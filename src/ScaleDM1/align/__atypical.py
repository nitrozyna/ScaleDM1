from __future__ import division

##
##Imports
import os
import re
import pysam
import random
import difflib
import subprocess
import numpy as np
import logging as log
from heapq import nlargest
from operator import itemgetter
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
		self.subsample_flag = sequencepair_object.get_subsampleflag()
		self.subsample_assembly = None
		self.subsample_index = None
		self.assembly_object = None
		self.present_references = None
		self.assembly_targets = None
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
		primary_data, secondary_data = self.organise_atypicals()

		for allele_pair in [(primary_object, primary_data, 'PRI'), (secondary_object, secondary_data, 'SEC')]:
			obj = allele_pair[0]; dat = allele_pair[1]
			obj.set_header(allele_pair[2])
			obj.set_referencelabel(dat.get('Reference'))
			obj.set_originalreference(dat.get('OriginalReference'))
			obj.set_totalreads(dat.get('TotalReads'))
			obj.set_ctgval(dat.get('EstimatedCTG'))
		sequencepair_object.set_primary_allele(primary_object)
		sequencepair_object.set_secondary_allele(secondary_object)

	def process_assembly(self):
		"""
        Function which processed the input SAM file for peak identification.
        Determinines the number of total reads present.
        Read file into PySAM object. Process further
        :param self:
        :return: None
        """

		##
		## Determine number of reads - for subsampling float
		## Use awk to read samtools idxstats output (get total read count)
		awk = ['awk', ' {i+=$3} END {print i}']
		count_process = subprocess.Popen(['samtools', 'idxstats', self.sorted_assembly], stdout=subprocess.PIPE)
		awk_process = subprocess.Popen(awk, stdin=count_process.stdout, stdout=subprocess.PIPE)
		count_process.wait();
		awk_process.wait();
		awk_output = int(awk_process.communicate()[0])
		if awk_output > 20000:
			subsample_float = 0.25
		elif 20000 > awk_output > 10000:
			subsample_float = 0.2
		else:
			subsample_float = 0.15
		self.sequencepair_object.set_subsampled_fqcount(awk_output)

		##
		## Subsample reads
		## Index the subsampled assembly
		if not self.sequencepair_object.get_broadflag():
			self.sequencepair_object.set_subsampleflag(subsample_float)
			self.sequencepair_object.set_automatic_DSPsubsample(True)
			self.subsample_assembly = os.path.join(self.sequence_path, 'subsample.sam')
			self.subsample_index = os.path.join(self.sequence_path, 'subsample.sam.bai')
			assem_obj = open(self.subsample_assembly, 'w')
			subsample_process = subprocess.Popen(
				['samtools', 'view', '-s', str(subsample_float), '-b', self.sorted_assembly], stdout=assem_obj)
			subsample_process.wait();
			assem_obj.close()
			index_process = subprocess.Popen(['samtools', 'index', self.subsample_assembly]);
			index_process.wait()
		else:
			self.subsample_assembly = self.sorted_assembly

		##
		## Load into object, determine references to investigate
		self.assembly_object = pysam.AlignmentFile(self.subsample_assembly, 'rb')
		self.present_references = self.assembly_object.references
		assembly_refdat = []
		for reference in self.present_references:
			reference_tuple = (reference, self.assembly_object.count(reference))
			if reference_tuple[1] == 0:
				pass
			else:
				assembly_refdat.append(reference_tuple)

		##
		## Assign our target references (Top 3 sorted references, sorted by read count)
		self.assembly_targets = sorted(assembly_refdat, key=itemgetter(1), reverse=True)[0:3]


		##
		## Check for a mal-aligned sample
		fail_score = 0
		for target in self.assembly_targets:
			if target[1] < 20:
				fail_score += 1
		if fail_score != 0:
			log.warning('{}{}{}{}'.format(clr.red, 'sdm1__ ', clr.end,
										  'Alignment contains too few reads. Cannot guarantee precision.'))
			self.alignment_warning = True
			self.sequencepair_object.set_alignmentwarning(self.alignment_warning)

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

			## Counts of reads
			ref_ctg = []
			##
			##
			## For every read in this reference, get the aligned sequence
			## Split into triplet sliding window list, remove any triplets that are < 3
			for read in self.assembly_object.fetch(reference=investigation[0]):
				target_sequence = read.query_alignment_sequence
				sequence_windows = [target_sequence[i:i + 3] for i in range(0, len(target_sequence), 3)]
				sequence_windows = [x for x in sequence_windows if len(x) == 3]

				## Doing the same idea of masks as Alastair just in case, you never know.
				ctg_masks = ['CTG', 'TGC', 'GCT']
				##
				## CTG Masking
				## Sort all list of tuples by length of second element (repeat tract length)
				ctg_tracts = []
				try:
					for mask in ctg_masks: ctg_tracts.append((mask, self.get_repeat_tract(sequence_windows, mask)))
					ctg_tract = sorted(ctg_tracts, key=lambda a: len(a[1]), reverse=True)[0][1]
				except IndexError:
					continue
				##
				## Add length to reference-run
				ref_ctg.append(len(ctg_tract))

			##
			## Calculate the presence of a 'state' of reference
			est_ctg = Counter(ref_ctg).most_common()[0][0]

			reference_dictionary = {'TotalReads': investigation[1],
									'EstimatedCTG': est_ctg}

			##
			## Append results to reference label
			self.atypical_info[investigation[0]] = reference_dictionary

		if not self.sequencepair_object.get_broadflag():
			os.remove(self.subsample_assembly)
			os.remove(self.subsample_index)

	def get_repeat_tract(self, triplet_input, mask):

		##
		## Score the entire read against the current mask
		current_tract = []
		for split in triplet_input:
			curr_score = self.similar(split, mask)
			current_tract.append((split, curr_score))

		##
		## Anchors
		region_start = None; region_end = None
		## Find the beginning of the CTG tract..
		## assuming streak of 3, confidence high in real start
		for i in range(0, len(current_tract)):
			try:
				if current_tract[i][1] == 1.0:
					if not region_start:
						if current_tract[i + 1][1] == 1.0 and current_tract[i + 2][1] == 1.0:
							region_start = i
				if current_tract[i][1] == 1.0:
					region_end = i
			except IndexError:
				pass

		##
		## If typeerror (i.e. one of the regions was None.. no start was found)
		## return empty list as there is no repeat tract for this mask
		try:
			first_pass_range = range(region_start, region_end + 1)
		except TypeError:
			return []

		##
		## Loop over rough range, remove items where n-1,n+1 and n+2 are not good matches for current mask
		for j in first_pass_range:
			if not current_tract[j][1] == 1.0:
				sub_score = 0
				try:
					for sub_check in [current_tract[j - 1], current_tract[j + 1], current_tract[j + 2]]:
						if sub_check[1] == 1.0: sub_score += 1
				except IndexError:
					pass
				if sub_score != 3: first_pass_range = [x for x in first_pass_range if x != j]

		##
		## Some downstream matches may exist still so..
		## remove anything outside of >1 streak in pass
		diff = 0;
		flagged_idx = 0
		for k in range(0, len(first_pass_range)):
			try:
				diff = abs(first_pass_range[k + 1] - first_pass_range[k])
			except IndexError:
				pass
			if diff > 1 and flagged_idx == 0: flagged_idx = first_pass_range[k] + 1
		for index in first_pass_range:
			if flagged_idx != 0 and index > flagged_idx:
				first_pass_range = [x for x in first_pass_range if x != index]

		##
		## Return list to call
		return first_pass_range

	def organise_atypicals(self):

		##
		## Constructs
		sorted_info = sorted(self.atypical_info.iteritems(), key=lambda (x, y): y['TotalReads'],reverse=True)
		if len(sorted_info) != 3: raise IndexError('< 3 references in sorted top; alignment failure?')

		sub_diff = float(abs(sorted_info[1][1]['TotalReads'] - sorted_info[2][1]['TotalReads']))
		sub_drop = float(sub_diff / sorted_info[1][1]['TotalReads'])

		## Top1 always used
		primary_allele = sorted_info[0][1]; primary_allele['Reference'] = sorted_info[0][0]
		secondary_allele = None

		##
		## CTG matches between #2/#3, potential peak skew
		##TODO lmao this is fucking horrible
		##TODO refactor this please
		## check #2 vs #3
		for val in [sorted_info[1], sorted_info[2]]:
			top1_reads = primary_allele['TotalReads']; curr_reads = val[1].get('TotalReads')
			read_drop = abs(top1_reads - curr_reads) / top1_reads
			if val[1].get('EstimatedCTG') != primary_allele['EstimatedCTG']: #TODO check if it's correct
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
				top1_top3_dist = abs(sorted_info[0][1]['EstimatedCTG'] - sorted_info[2][1]['EstimatedCTG'])
				top1_top2_dist = abs(sorted_info[0][1]['EstimatedCTG'] - sorted_info[1][1]['EstimatedCTG'])
				top2_top3_dist = abs(sorted_info[1][1]['EstimatedCTG'] - sorted_info[2][1]['EstimatedCTG'])
				if read_drop >= 0.65:
					if top2_top3_dist == 1:
						top2_ctg = sorted_info[1][1]['EstimatedCTG']; top3_ctg = sorted_info[2][1]['EstimatedCTG']
						##
						## Diminished Peak (Top2)
						if top2_ctg > top3_ctg:
							if np.isclose([sub_drop], [0.5], atol=0.1):
								if not np.isclose([primary_allele['EstimatedCTG']], [top2_ctg], atol=5):
									secondary_allele = sorted_info[1][1]
									secondary_allele['Reference'] = sorted_info[1][0]
									break
								else:
									secondary_allele = primary_allele.copy()
									break
							else:
								secondary_allele = primary_allele.copy()
								break
						##
						## Diminished peak (Top3)
						elif top3_ctg > top2_ctg:
							if np.isclose([sub_drop], [0.2], atol=0.2):
								if not np.isclose([primary_allele['EstimatedCTG']], [top3_ctg], atol=5):
									secondary_allele = sorted_info[1][1]
									secondary_allele['Reference'] = sorted_info[1][0]
									break
								else:
									secondary_allele = primary_allele.copy()
									break
							else:
								secondary_allele = primary_allele.copy()
								break
					else:
						secondary_allele = sorted_info[1][1]
						secondary_allele['Reference'] = sorted_info[1][0]

					##
					## Legit peak (not diminished or homozyg)
				elif 0.0 < read_drop < 0.64:
					if not top1_top2_dist == 1:
						secondary_allele = sorted_info[1][1]
						secondary_allele['Reference'] = sorted_info[1][0]
						break
					else:
						if np.isclose([sub_drop], [1.5], atol=0.25):
							secondary_allele = sorted_info[1][1]
							secondary_allele['Reference'] = sorted_info[1][0]
							break
						else:
							secondary_allele = sorted_info[2][1]
							secondary_allele['Reference'] = sorted_info[2][0]
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
		## For each of the alleles we've determined..
		## Get intervening lengths, create accurate genotype string
		for allele in [primary_allele, secondary_allele]:

			new_genotype = self.create_genotype_label(allele)
			allele['OriginalReference'] = allele['Reference']
			allele['Reference'] = new_genotype
		print primary_allele, secondary_allele
		return primary_allele, secondary_allele

	def create_genotype_label(self,input_reference):

		genotype_label = '{}'.format(input_reference['EstimatedCTG'])

		return genotype_label

	@staticmethod
	def similar(seq1, seq2):
		return difflib.SequenceMatcher(a=seq1.lower(), b=seq2.lower()).ratio()
