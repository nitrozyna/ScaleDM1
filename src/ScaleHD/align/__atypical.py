from __future__ import division

##
##Imports
import os
import regex
import pysam
import difflib
import subprocess
import numpy as np
import logging as log
from collections import Counter
from ..__backend import Colour as clr
from ..__allelecontainer import IndividualAllele

class ScanAtypical:
	def __init__(self, sequencepair_object, instance_params):
		"""
		Class which utilises Digital Signal Processing to determine the repeat tract structure of the current sample.
		General overview:
			Subsample aligned SAM file (from __alignment.py) to increase speed of function
			Loop over the references with the top 3 highest number of aligned reads (only ones relevant to genotyping)
			For each read in that reference, scan for regions using rotating masks
			Return regions, determine values
			Assign values to appropriate allele object class variables
			Return
		:param sequencepair_object: Object of the current Sequence pair being processed.. see __allelecontainer.py
		:param instance_params: Dictionary of config settings from the input XML document
		"""

		##
		## Variables for this class/assembly data
		self.sequencepair_object = sequencepair_object
		self.sequence_path = sequencepair_object.get_alignpath()
		self.sorted_assembly = sequencepair_object.get_fwassembly()
		self.instance_params = instance_params
		self.subsample_assembly = None
		self.subsample_index = None
		self.assembly_object = None
		self.present_references = None
		self.assembly_targets = None
		self.atypical_count = 0
		self.awk_output = 0
		self.atypical_info = {}
		self.alignment_warning = False

		##
		## Fill objects with data
		self.process_assembly()

		##
		## Run the scanning algorithm
		## Exception for (unexpected) EOF
		try: self.scan_reference_reads()
		except StopIteration: self.assembly_object.close()

		##
		## Turn results into objects
		primary_object = IndividualAllele(); secondary_object = IndividualAllele()
		primary_data, secondary_data, atypical_count = self.organise_atypicals()

		sequencepair_object.set_atypical_count(atypical_count)
		for allele_pair in [(primary_object, primary_data, 'PRI'), (secondary_object, secondary_data, 'SEC')]:
			obj = allele_pair[0]; dat = allele_pair[1]
			obj.set_header(allele_pair[2])
			obj.set_allelestatus(dat.get('Status'))
			obj.set_referencelabel(dat.get('Reference'))
			obj.set_originalreference(dat.get('OriginalReference'))
			obj.set_totalreads(dat.get('TotalReads'))
			obj.set_typicalreads(dat.get('TypicalCount'))
			obj.set_typicalpcnt(dat.get('TypicalPcnt'))
			obj.set_atypicalreads(dat.get('AtypicalCount'))
			obj.set_atypicalpcnt(dat.get('AtypicalPcnt'))
			obj.set_fiveprime(dat.get('5PFlank'))
			obj.set_cagval(dat.get('EstimatedCAG'))
			obj.set_intervening(dat.get('InterveningSequence'))
			obj.set_caacagval(dat.get('EstimatedCAACAG'))
			obj.set_ccgccaval(dat.get('EstimatedCCGCCA'))
			obj.set_ccgval(dat.get('EstimatedCCG'))
			obj.set_cctval(dat.get('EstimatedCCT'))
			obj.set_threeprime(dat.get('3PFlank'))
		sequencepair_object.set_primary_allele(primary_object)
		sequencepair_object.set_secondary_allele(secondary_object)

		##
		## Generate an atypical report for writing
		self.atypical_report = os.path.join(self.sequence_path, 'AtypicalReport.txt')
		report_file = open(self.atypical_report, 'w')
		report_file.write('{}{}\n{}{}\n{}{}\n{}{}'.format('Primary Allele: ', primary_object.get_reflabel(),
			'Primary Original: ', primary_object.get_originalreference(),
			'Secondary Allele: ', secondary_object.get_reflabel(),
			'Secondary Original: ', secondary_object.get_originalreference()))
		report_file.close()

	def process_assembly(self):
		"""
		Function which processes the input SAM for atypical scanning.
		Determine the number of total reads present (for subsampling).
		If the user specified to use --boost, or there are more than 20k reads present, we subsample.
		Read file into PySAM object.. process further
		:return: None
		"""

		##
		## Determine number of reads - for subsampling float
		## Use awk to read samtools idxstats output (get total read count)
		awk = ['awk', ' {i+=$3} END {print i}']
		count_process = subprocess.Popen(['samtools','idxstats', self.sorted_assembly], stdout=subprocess.PIPE)
		awk_process = subprocess.Popen(awk, stdin=count_process.stdout, stdout=subprocess.PIPE)
		count_process.wait(); awk_process.wait(); awk_output = int(awk_process.communicate()[0])
		if awk_output > 20000: subsample_float = 0.05
		elif 20000 > awk_output > 10000: subsample_float = 0.1
		else: subsample_float = 0.2
		self.sequencepair_object.set_totalseqreads(awk_output)

		##
		## Subsample reads
		## Index the subsampled assembly
		if self.sequencepair_object.get_boostflag() or awk_output > 20000:
			self.sequencepair_object.set_subsampleflag('0.05**')
			self.subsample_assembly = os.path.join(self.sequence_path,'subsample.sam')
			self.subsample_index = os.path.join(self.sequence_path,'subsample.sam.bai')
			assem_obj = open(self.subsample_assembly,'w')
			subsample_process = subprocess.Popen(['samtools','view','-s',str(subsample_float),'-b', self.sorted_assembly], stdout=assem_obj)
			subsample_process.wait(); assem_obj.close()
			index_process = subprocess.Popen(['samtools','index',self.subsample_assembly]); index_process.wait()
		##
		## If the user has specified a global subsampling flag, we do not subsample further (unless insane read count)
		if not self.sequencepair_object.get_boostflag():
			if awk_output > 20000: pass
			else: self.subsample_assembly = self.sorted_assembly

		##
		## Load into object, determine references to investigate
		self.assembly_object = pysam.AlignmentFile(self.subsample_assembly, 'rb')
		self.present_references = self.assembly_object.references
		assembly_refdat = []
		for reference in self.present_references:
			reference_tuple = (reference, self.assembly_object.count(reference))
			if reference_tuple[1] == 0: pass
			else: assembly_refdat.append(reference_tuple)

		##
		## Assign our target references (Top 3 sorted references, sorted by read count)
		self.assembly_targets = sorted(assembly_refdat, key=lambda x:x[1], reverse=True)[0:3]

		##
		## Check for a mal-aligned sample
		fail_score = 0
		for target in self.assembly_targets:
			if target[1] < 20:
				fail_score += 1
		if fail_score != 0:
			log.warning('{}{}{}{}'.format(clr.red, 'shd__ ', clr.end, 'Alignment contains too few reads. Cannot guarantee precision.'))
			self.alignment_warning = True
			self.sequencepair_object.set_alignmentwarning(self.alignment_warning)

	@staticmethod
	def typical_rotation(input_string):

		"""
		Function to detect if an intervening sequence (typical structure) is located within all possible
		rotations of a given/derived intervening sequence. Easiest method is to double to target, and search.
		:param input_string:
		:return:
		"""

		##
		## Lengths of target strings
		size1 = len('CAACAGCCGCCA')
		size2 = len(input_string)
		temp = ''

		##
		## Size equality comparison
		if size1 != size2: return 0

		##
		## Duplicate string to encompass all possible rotations
		temp = 'CAACAGCCGCCA' + 'CAACAGCCGCCA'

		##
		## Now check STR2 is a substring of temp expansion
		## .count() returns number of occurrences of second string in temp
		if temp.count(input_string) > 0: return 1
		else: return 0

	def scan_reference_reads(self):

		"""
		Function which determines the literal repeat regions, ignoring misalignment issues.
		We loop over every 'investigation' from this assembly <- i.e. the top 3 reference (in terms of read count)
		Each read within each reference is then scanned, to determine the structure of said read.
		:return:
		"""


		##
		## Iterate over top 3 aligned references in this assembly
		## Fetch the reads aligned to the current reference
		for investigation in self.assembly_targets:
			reference_data = self.assembly_object.fetch(reference=investigation[0])

			##
			## Counts of atypical/typical reads
			typical_count = 0; atypical_count = 0; reference_atypicals = []; fp_flanks = []; tp_flanks = []
			ref_cag = []; ref_ccg = []; ref_cct = []

			##
			## For every read in this reference, get the aligned sequence
			## Split into triplet sliding window list, remove any triplets that are < 3
			for read in reference_data:
				target_sequence = read.query_alignment_sequence
				sequence_windows = [target_sequence[i:i + 3] for i in range(0, len(target_sequence), 3)]
				sequence_windows = [x for x in sequence_windows if len(x)==3]

				##
				## Get repeat regions for CAG and CCG; based on similarity mask scores for the current window
				## Any regions returned that are (idx > end_of_region) are truncated
				## CAG and CCG repeat region index list combined
				cag_masks = ['CAG', 'AGC', 'GCA']
				ccg_masks = ['CCG', 'CGC', 'GCC']
				cct_masks = ['CCT', 'CTC', 'TCC']

				##
				## CAG/CCG Masking
				## Sort all list of tuples by length of second element (repeat tract length)
				## Select first item as 'true' tract, then calculate intervening sequence length
				cag_tracts = []; ccg_tracts = []; cct_tracts = []
				try:
					for mask in cag_masks: cag_tracts.append((mask, self.get_repeat_tract(sequence_windows, mask)))
					for mask in ccg_masks: ccg_tracts.append((mask, self.get_repeat_tract(sequence_windows, mask)))
					cag_tract = sorted(cag_tracts, key=lambda a: len(a[1]), reverse=True)[0][1]
					ccg_tract = sorted(ccg_tracts, key=lambda a: len(a[1]), reverse=True)[0][1]

					##
					## CCT Masking/Intervening calculation
					intervene_string = ''; fp_flank_string = ''; tp_flank_string = ''
					for mask in cct_masks: cct_tracts.append((mask, self.get_cct_tract(sequence_windows, mask, ccg_tract[-1])))
					cct_tract = sorted(cct_tracts, key=lambda a: len(a[1]), reverse=True)[0][1]
					intervene_range = range(cag_tract[-1]+1, ccg_tract[0])
					fp_flank_range = range(0, cag_tract[0]-1)
					tp_flank_range = range(cct_tract[-1]+1, len(sequence_windows))
				except IndexError:
					continue

				##
				## Add length to reference-run
				ref_cag.append(len(cag_tract))
				ref_ccg.append(len(ccg_tract))
				ref_cct.append(len(cct_tract))

				##
				## Count fp flank occurrences
				for i in range(0, len(sequence_windows)):
					if i in fp_flank_range:
						fp_flank_string += str(sequence_windows[i])
				fp_flanks.append(fp_flank_string)

				##
				## Count tp flank occurrences
				for i in range (0, len(sequence_windows)):
					if i in tp_flank_range:
						tp_flank_string += str(sequence_windows[i])
				tp_flanks.append(tp_flank_string)

				##
				## Atypical Detection
				for i in range(0, len(sequence_windows)):
					if i in intervene_range:
						intervene_string += str(sequence_windows[i])
				if self.rotation_check('CAACAGCCGCCA', intervene_string):
					intervene_string = 'CAACAGCCGCCA'
				if intervene_string != 'CAACAGCCGCCA':
					atypical_count += 1
					reference_atypicals.append(intervene_string)
				else:
					typical_count += 1

			##
			## Calculate the presence of each 'state' of reference
			ref_typical = format(((typical_count / investigation[1]) * 100), '.2f')
			ref_atypical = format(((atypical_count / investigation[1]) * 100), '.2f')
			est_cag = Counter(ref_cag).most_common()[0][0]
			est_ccg = Counter(ref_ccg).most_common()[0][0]
			est_cct = Counter(ref_cct).most_common()[0][0]

			##
			## Determine most frequent intervening sequence
			atypical_population = Counter(reference_atypicals).most_common()
			fp_flank_population = Counter(fp_flanks).most_common()
			tp_flank_population = Counter(tp_flanks).most_common()

			if len(atypical_population) == 0: atypical_population = [['CAACAGCCGCCA']]
			reference_dictionary = {'TotalReads':investigation[1],
									'TypicalCount': typical_count,
									'TypicalPcnt': ref_typical,
									'AtypicalCount': atypical_count,
									'AtypicalPcnt': ref_atypical,
									'Status':self.atypical_count,
									'5PFlank':fp_flank_population[0][0],
									'3PFlank':tp_flank_population[0][0],
									'EstimatedCAG': est_cag,
									'EstimatedCCG': est_ccg,
									'EstimatedCCT': est_cct,
									'InterveningSequence': atypical_population[0][0]}

			if atypical_count > typical_count:
				self.atypical_count += 1
				reference_dictionary['Status'] = 'Atypical'
			elif est_cct != 2:
				self.atypical_count += 1
				reference_dictionary['Status'] = 'Atypical'
			else:
				reference_dictionary['Status'] = 'Typical'
				reference_dictionary['InterveningSequence'] = 'CAACAGCCGCCA'

			##
			## If the intervening is longer in #2, assume poor sequencing in #1 and use #2
			try:
				if len(atypical_population[0][0]) < len(atypical_population[1][0]):
					if reference_dictionary['Status'] == 'Typical':
						reference_dictionary['InterveningSequence'] = max([atypical_population[0][0],atypical_population[1][0]], key=len)
			except IndexError:
				reference_dictionary['InterveningSequence'] = atypical_population[0][0]

			##
			## Check for mismatch just before intervening sequence
			try:
				top_hit = atypical_population[0][1]; second_hit = atypical_population[1][1]
				diff = ((top_hit-second_hit)/top_hit)*100
				if diff < 30.00:
					if len(atypical_population[0][0]) == 15:
						if np.isclose(self.similar('CAG', atypical_population[0][0][0:3]), [0.66], atol=0.1):
							reference_dictionary['InterveningSequence'] = 'CAACAGCCGCCA'
			except IndexError:
				pass

			##
			## Append results to reference label
			self.atypical_info[investigation[0]] = reference_dictionary

		if self.sequencepair_object.get_boostflag() or self.sequencepair_object.get_subsampleflag() == '0.05**':
			os.remove(self.subsample_assembly)
			os.remove(self.subsample_index)

	def get_repeat_tract(self, triplet_input, mask):

		##
		## Score the entire read against the current mask
		current_tract = []
		for split in triplet_input:
			curr_score = self.similar(split,mask)
			current_tract.append((split,curr_score))

		##
		## Anchors
		region_start = None; region_end = None
		## Find the beginning of the CAG tract..
		## assuming streak of 3, confidence high in real start
		for i in range(0, len(current_tract)):
			try:
				if current_tract[i][1] == 1.0:
					if not region_start:
						if current_tract[i+1][1] == 1.0 and current_tract[i+2][1] == 1.0:
							region_start = i
				if current_tract[i][1] == 1.0:
					region_end = i
			except IndexError:
				pass

		##
		## If typeerror (i.e. one of the regions was None.. no start was found)
		## return empty list as there is no repeat tract for this mask
		try: first_pass_range = range(region_start, region_end+1)
		except TypeError: return []

		##
		## Loop over rough range, remove items where n-1,n+1 and n+2 are not good matches for current mask
		for j in first_pass_range:
			if not current_tract[j][1] == 1.0:
				sub_score = 0
				try:
					for sub_check in [current_tract[j-1], current_tract[j+1], current_tract[j+2]]:
						if sub_check[1] == 1.0: sub_score += 1
				except IndexError:
					pass
				if sub_score != 3: first_pass_range = [x for x in first_pass_range if x!=j]

		##
		## Some downstream matches may exist still so..
		## remove anything outside of >1 streak in pass
		diff = 0; flagged_idx = 0
		for k in range(0, len(first_pass_range)):
			try: diff = abs(first_pass_range[k+1]-first_pass_range[k])
			except IndexError: pass
			if diff > 1 and flagged_idx == 0: flagged_idx = first_pass_range[k]+1
		for index in first_pass_range:
			if flagged_idx != 0 and index > flagged_idx:
				first_pass_range = [x for x in first_pass_range if x!= index]

		##
		## Return list to call
		return first_pass_range

	def get_cct_tract(self, triplet_input, mask, anchor):

		##
		## Get all triplets after the end of the CCG tract (anchor)
		post_anchor = []
		for i in range(0, len(triplet_input)):
			if i > anchor: post_anchor.append((i, triplet_input[i]))

		##
		## If similarity matches the current mask, add that index to tract
		cct_tract = []
		for item in post_anchor:
			if self.similar(mask, item[1]) == 1.0:
				cct_tract.append(item[0])

		##
		## Remove indexes in tract list if difference between indexes > 1 (gaps dont happen in cct)
		diff = 0; flagged_idx = 0
		for i in range(0, len(cct_tract)):
			try: diff = abs(cct_tract[i+1]-cct_tract[i])
			except IndexError: pass
			if diff > 1 and flagged_idx == 0: flagged_idx = cct_tract[i]+1
		for index in cct_tract:
			if flagged_idx!=0 and index>flagged_idx:
				cct_tract = [x for x in cct_tract if x!=index]

		##
		## Return
		return cct_tract

	def organise_atypicals(self):

		##
		## Constructs
		sorted_info = sorted(self.atypical_info.iteritems(), key=lambda (x, y): y['TotalReads'], reverse=True)
		if len(sorted_info) != 3: raise IndexError('< 3 references in sorted top; alignment failure?')

		##
		## Check % dropoff in read count between #2 and #3
		alpha_diff = float(abs(sorted_info[0][1]['TotalReads'] - sorted_info[1][1]['TotalReads']))
		beta_diff = float(abs(sorted_info[0][1]['TotalReads'] - sorted_info[2][1]['TotalReads']))
		sub_diff = float(abs(sorted_info[1][1]['TotalReads'] - sorted_info[2][1]['TotalReads']))
		alpha_drop = float(alpha_diff / sorted_info[0][1]['TotalReads'])
		beta_drop = float(beta_diff / sorted_info[0][1]['TotalReads'])
		sub_drop = float(sub_diff / sorted_info[1][1]['TotalReads'])

		## Top1 always used
		primary_allele = sorted_info[0][1]; primary_allele['Reference'] = sorted_info[0][0]
		secondary_allele = None

		##
		## CCG matches between #2/#3, potential peak skew
		if sorted_info[1][1]['EstimatedCCG'] == sorted_info[2][1]['EstimatedCCG']:
			##
			## check #2 and #3 vs CAG(#1)
			for val in [sorted_info[1], sorted_info[2]]:
				top1_reads = primary_allele['TotalReads']; curr_reads = val[1].get('TotalReads')
				read_drop = abs(top1_reads-curr_reads)/top1_reads

				if val[1].get('EstimatedCCG') != primary_allele['EstimatedCCG']:
					if read_drop >= 0.40:
						if sub_drop <= 0.25:
							secondary_allele = sorted_info[2][1]
							secondary_allele['Reference'] = sorted_info[2][0]
							break
						else:
							secondary_allele = sorted_info[1][1]
							secondary_allele['Reference'] = sorted_info[1][0]
							break
					else:
						if sub_drop <= 0.25:
							secondary_allele = sorted_info[1][1]
							secondary_allele['Reference'] = sorted_info[1][0]
							break

				##
				## Secondary allele unassigned, perhaps homzoygous haplotype
				if not secondary_allele:
					top1_top3_dist = abs(sorted_info[0][1]['EstimatedCAG']-sorted_info[2][1]['EstimatedCAG'])
					top2_top3_dist = abs(sorted_info[1][1]['EstimatedCAG']-sorted_info[2][1]['EstimatedCAG'])
					if read_drop >= 0.65 and top2_top3_dist == 1:
						secondary_allele = primary_allele
						break
					elif 0 < read_drop < 0.64:
						secondary_allele = sorted_info[1][1]
						secondary_allele['Reference'] = sorted_info[1][0]
						break
					elif top2_top3_dist >= 2:
						if not top1_top3_dist == 1:
							secondary_allele = sorted_info[2][1]
							secondary_allele['Reference'] = sorted_info[2][0]
							break
						else:
							secondary_allele = sorted_info[1][1]
							secondary_allele['Reference'] = sorted_info[1][0]
							break
					else:
						secondary_allele = sorted_info[1][1]
						secondary_allele['Reference'] = sorted_info[1][0]
						break
		##
		## CCG mismatch between #2/#3, no potential peak skew
		else:
			if sorted_info[0][1]['EstimatedCCG'] == sorted_info[1][1]['EstimatedCCG']:
				if np.isclose([sorted_info[0][1]['EstimatedCAG']], [sorted_info[1][1]['EstimatedCAG']],atol=1):
					if sorted_info[0][1]['EstimatedCCG'] != sorted_info[2][1]['EstimatedCCG']:
						secondary_allele = sorted_info[2][1]
						secondary_allele['Reference'] = sorted_info[2][0]
			elif alpha_drop >= 0.65 and beta_drop >= 0.80:
				secondary_allele = primary_allele
			elif beta_drop >= 0.20:
				secondary_allele = sorted_info[1][1]
				secondary_allele['Reference'] = sorted_info[1][0]

		##
		## For each of the alleles we've determined..
		## Get intervening lengths, create accurate genotype string
		atypical_count = 0
		for allele in [primary_allele, secondary_allele]:
			new_genotype, caacag_count, ccgcca_count = self.create_genotype_label(allele)
			allele['OriginalReference'] = allele['Reference']
			allele['Reference'] = new_genotype
			allele['EstimatedCAACAG'] = caacag_count
			allele['EstimatedCCGCCA'] = ccgcca_count
			if allele['Status'] == 'Atypical': atypical_count += 1

		return primary_allele, secondary_allele, atypical_count

	def create_genotype_label(self, input_reference):

		intervening = input_reference['InterveningSequence']
		intervening_freq = Counter(list((intervening[0 + i:6 + i] for i in range(0, len(intervening), 6)))).items()
		caacag_count = 0; ccgcca_count = 0
		caacag_flag = False; ccgcca_flag = False

		##
		## TODO fix this dumpster garbage shit
		## TODO oh my god it's so ugly
		if len(intervening_freq) < 1: caacag_count = 1; ccgcca_count = 1
		if len(intervening_freq) < 2: caacag_count = 1; ccgcca_count = 1

		##
		## Check CAACAG
		try:
			caacag_freq = intervening_freq[0]
			if self.rotation_check('CAACAG', caacag_freq[0]):
				caacag_count = caacag_freq[1]
			else:
				if input_reference['Status'] == 'Typical':
					caacag_count = 1; ccgcca_count = 1
		except IndexError:
			caacag_flag = True

		##
		## Check CCGCCA
		try:
			ccgcca_freq = intervening_freq[1]
			if self.rotation_check('CCGCCA', ccgcca_freq[0]):
				ccgcca_count = ccgcca_freq[1]
			else:
				if input_reference['Status'] == 'Typical':
					caacag_count = 1; ccgcca_count = 1
		except IndexError:
			ccgcca_flag = True

		##
		## Parse flags in event of error
		if caacag_flag: caacag_count = 0; ccgcca_count = 0
		if ccgcca_flag:
			if not caacag_flag:
				caacag_count = 1; ccgcca_count = 0
			else:
				caacag_count = 0; ccgcca_count = 0

		##
		## Safety check
		if input_reference['Status'] == 'Typical' and (caacag_count != 1 or ccgcca_count != 1):
			caacag_count = 1; ccgcca_count = 1

		genotype_label = '{}_{}_{}_{}_{}'.format(input_reference['EstimatedCAG'], caacag_count, ccgcca_count,
												 input_reference['EstimatedCCG'], input_reference['EstimatedCCT'])
		return genotype_label, caacag_count, ccgcca_count

	def get_atypicalreport(self):
		return self.atypical_report

	@staticmethod
	def rotation_check(string1, string2):
		size1 = len(string1)
		size2 = len(string2)

		# Check if sizes of two strings are same
		if size1 != size2: return 0

		# Create a temp string with value str1.str1
		temp = string1 + string1

		# Now check if str2 is a substring of temp (with s = 1 mismatch)
		rotation_match = regex.findall(r"(?:" + string2 + "){s<=1}", temp, regex.BESTMATCH)

		if len(rotation_match) > 0:
			return 1
		else:
			return 0

	@staticmethod
	def similar(seq1, seq2):
		return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio()