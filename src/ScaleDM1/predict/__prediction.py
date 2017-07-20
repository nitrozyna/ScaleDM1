from __future__ import division

#/usr/bin/python
__version__ = 0.01
__author__ = 'alastair.maxwell@glasgow.ac.uk'

##
## Generic imports
import os
import csv
import PyPDF2
import warnings
import peakutils
import matplotlib
import numpy as np
import logging as log
matplotlib.use('Agg')
from sklearn import svm
import prettyplotlib as ppl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from reportlab.pdfgen import canvas
from peakutils.plot import plot as pplot
from sklearn.multiclass import OutputCodeClassifier

##
## Backend Junk
from ..__backend import DataLoader
from ..__backend import Colour as clr

class AlleleGenotyping:
	def __init__(self, sequencepair_object, instance_params, training_data):

		##
		## Allele objects and instance data
		self.sequencepair_object = sequencepair_object
		self.instance_params = instance_params
		self.training_data = training_data
		self.allele_report = ''
		self.warning_triggered = False

		##
		## Constructs that will be updated with each allele process
		self.classifier, self.encoder = self.build_zygosity_model()
		self.allele_flags = {}; self.forward_distribution = None; self.reverse_distribution = None
		self.forward_aggregate = None; self.reverse_aggregate = None
		self.expected_zygstate = None; self.zygosity_state = None
		self.pass_vld = True; self.ctg_sum = []

		##
		## Genotype!
		if not self.allele_validation(): raise Exception('Allele(s) failed validation. Cannot genotype..')
		if not self.determine_ctg(): raise Exception('CTG Genotyping failure. Cannot genotype..')
		if not self.genotype_validation(): raise Exception('Genotype failed validation. Cannot genotype..')
		if not self.inspect_peaks():
			log.warn('{}{}{}{}'.format(clr.red, 'sdm1__ ', clr.end, '1+ allele(s) failed peak validation. Precision not guaranteed.'))
			self.warning_triggered = True
			self.sequencepair_object.set_peakinspection_warning(True)
		self.render_graphs()
		self.calculate_score()
		self.set_report()

	def build_zygosity_model(self):
		"""
		Function to build a SVM (wrapped into OvO class) for determining CTG zygosity
		:return: svm object wrapped into OvO, class-label hash-encoder object
		"""

		##
		## Classifier object and relevant parameters for our CCG prediction
		svc_object = svm.LinearSVC(C=1.0, loss='ovr', penalty='l2', dual=False,
								   tol=1e-4, multi_class='crammer_singer', fit_intercept=True,
								   intercept_scaling=1, verbose=0, random_state=0, max_iter=-1)

		##
		## Take raw training data (CTG zygosity data) into DataLoader model object
		traindat_ctg_collapsed = self.training_data['CollapsedCTGZygosity']
		traindat_descriptionfi = self.training_data['GenericDescriptor']
		traindat_model = DataLoader(traindat_ctg_collapsed, traindat_descriptionfi).load_model()

		##
		## Model data fitting to SVM
		X = preprocessing.normalize(traindat_model.DATA)
		Y = traindat_model.TARGET
		ovo_svc = OutputCodeClassifier(svc_object, code_size=2, random_state=0).fit(X, Y)
		encoder = traindat_model.ENCDR

		##
		## Return the fitted OvO(SVM) and Encoder
		return ovo_svc, encoder

	def predict_zygstate(self):
		"""
		Function which takes the newly collapsed CTG distribution and executes SVM prediction
		to determine the zygosity state of this sample's CTG value(s). Data is reshaped
		and normalised to ensure more reliable results. A check is executed between the results of
		forward and reverse zygosity; if a match, great; if not, not explicitly bad but inform user.
		:return: zygosity[2:-2] (trimming unrequired characters)
		"""

		##
		## Reshape the input distribution so SKL doesn't complain about 1D vectors
		## Normalise data in addition; cast to float64 for this to be permitted
		forward_reshape = preprocessing.normalize(np.float64(self.forward_aggregate.reshape(1, -1)))
		reverse_reshape = preprocessing.normalize(np.float64(self.reverse_aggregate.reshape(1, -1)))

		##
		## Predict the zygstate of these reshapen, noramlised 20D CCG arrays using SVM object earlier
		## Results from self.classifier are #encoded; so convert with our self.encoder.inverse_transform
		forward_zygstate = str(self.encoder.inverse_transform(self.classifier.predict(forward_reshape)))
		reverse_zygstate = str(self.encoder.inverse_transform(self.classifier.predict(reverse_reshape)))


		##
		## We only particularly care about the forward zygosity (CTG reads are higher quality in forward data)
		## However, for a QoL metric, compare fw/rv results. If match, good! If not, doesn't matter!
		if not forward_zygstate == reverse_zygstate:
			self.allele_flags['CCGZygDisconnect'] = True
		else:
			self.allele_flags['CCGZyg_disconnect'] = False

		return reverse_zygstate[2:-2]


	def index_inspector(self, index_inspection_count):

		major = max(self.reverse_aggregate)
		majoridx = np.where(self.reverse_aggregate == major)[0][0]
		minor = max(n for n in self.reverse_aggregate if n != major)
		minoridx = np.where(self.reverse_aggregate == minor)[0][0]

		if index_inspection_count == 2:
			return [(major, majoridx), (minor, minoridx)]
		if index_inspection_count == 1:
			return [(major, majoridx)]


	@staticmethod
	def scrape_distro(distributionfi):
		"""
		Function to take the aligned read-count distribution from CSV into a numpy array
		:param distributionfi:
		:return: np.array(data_from_csv_file)
		"""

		##
		## Open CSV file with information within; append to temp list
		## Scrape information, cast to np.array(), return
		placeholder_array = []
		with open(distributionfi) as dfi:
			source = csv.reader(dfi, delimiter=',')
			next(source)  # skip header
			for row in source:
				placeholder_array.append(int(row[2]))
			dfi.close()
		unlabelled_distro = np.array(placeholder_array)
		return unlabelled_distro


	@staticmethod
	def pad_distribution(distribution_array, allele_object):

		local_index = np.where(distribution_array == max(distribution_array))[0][0]
		local_rightpad = len(distribution_array) - local_index
		global_index = allele_object.get_ctg() - 1
		left_buffer = abs(local_index - global_index)
		right_buffer = abs(20 - global_index) - local_rightpad
		left_pad = np.asarray([0] * left_buffer)
		right_pad = np.asarray([0] * right_buffer)
		left_aug = np.concatenate((left_pad, distribution_array))
		right_aug = np.concatenate((left_aug, right_pad))

		return right_aug


	def close_check(self, allele, array, x, y, z, state=None):
		inner_pass = True
		if np.isclose(array, x, atol=y):
			if state == 'minus':
				allele.set_nminuswarninglevel(z)
				if z >= 5: inner_pass = False
			if state == 'plus':
				allele.set_npluswarninglevel(z)
				if z >= 5: inner_pass = False
		else:
			allele.set_nminuswarninglevel(0)
			allele.set_npluswarninglevel(0)
		self.pass_vld = inner_pass

	def peak_detection(self, allele_object, distro, peak_dist, triplet_stage):

		##
		## Status
		fail_state = False
		utilised_threshold = 0.50
		error_boundary = 0

		allele_object.set_ctgthreshold(utilised_threshold)
		if triplet_stage == "CTGHom" or triplet_stage == "CTGHet" or triplet_stage == "CTG": error_boundary = 1
		##TODO what happends when you do pass


		##
		## Look for peaks in our distribution
		peak_indexes = peakutils.indexes(distro, thres=utilised_threshold, min_dist=peak_dist)
		fixed_indexes = np.array(peak_indexes + 1)
		if not len(fixed_indexes) == error_boundary:
			if allele_object.get_ctg() in fixed_indexes:
				fixed_indexes = np.asarray([x for x in fixed_indexes if x == allele_object.get_ctg()])
			elif triplet_stage == 'CTG':
				if len(fixed_indexes) > 2 and not self.zygosity_state == 'HETERO':
					fixed_indexes = [np.where(distro == max(distro))[0][0] + 1]
					if self.zygosity_state == 'HOMO':
						self.sequencepair_object.set_svm_failure(False)
						pass
					else:
						self.sequencepair_object.set_svm_failure(True)
						self.sequencepair_object.set_alignmentwarning(True)

			else:
				fail_state = True


		return fail_state, fixed_indexes
		##
		## CCG, CAGHet (Hetero, Homo*, Homo+), CAGDim == 1 peak
		## CAGHom == 2 peaks
		allele_object.set_cagthreshold(utilised_threshold)
		if triplet_stage == 'CAGHom': error_boundary = 2
		else: error_boundary = 1

		##
		## Look for peaks in our distribution
		peak_indexes = peakutils.indexes(distro, thres=utilised_threshold, min_dist=peak_dist)
		fixed_indexes = np.array(peak_indexes + 1)
		if not len(fixed_indexes) == error_boundary:
			if triplet_stage == 'CAGHom' and (est_dist==1 or est_dist==0):
				pass
			elif allele_object.get_cag() in fixed_indexes:
				fixed_indexes = np.asarray([x for x in fixed_indexes if x == allele_object.get_cag()])
			elif triplet_stage == 'CCG':
				if len(fixed_indexes) > 2 and not self.zygosity_state == 'HETERO':
					fixed_indexes = [np.where(distro == max(distro))[0][0]+1]
					if self.zygosity_state == 'HOMO+' or self.zygosity_state == 'HOMO*':
						self.sequencepair_object.set_svm_failure(False)
						pass
					else:
						self.sequencepair_object.set_svm_failure(True)
						self.sequencepair_object.set_alignmentwarning(True)
			else:
				fail_state = True

		return fail_state, fixed_indexes

	def allele_validation(self):

		ctg_expectant = []
		##
		## For the two allele objects in this sample_pair
		for allele_object in [self.sequencepair_object.get_primaryallele(),
							  self.sequencepair_object.get_secondaryallele()]:
			print allele_object
			##
			## Assign read mapped percent if not present in allele
			if not allele_object.get_fwalnpcnt() and not allele_object.get_rvalnpcnt():
				allele_object.set_fwalnpcnt(self.sequencepair_object.get_fwalnpcnt())
				allele_object.set_rvalnpcnt(self.sequencepair_object.get_rvalnpcnt())

			##
			## Unlabelled distributions
			self.forward_distribution = self.scrape_distro(allele_object.get_fwdist())
			self.reverse_distribution = self.scrape_distro(allele_object.get_rvdist())
			allele_object.set_fwarray(self.forward_distribution)
			allele_object.set_rvarray(self.reverse_distribution)

			##
			## Distribution read count / Peak read count
			if allele_object.get_totalreads() < 1000:
				allele_object.set_distribution_readcount_warning(True)
				self.sequencepair_object.set_distribution_readcount_warning(True)

			##
			## If current allele obj's assembly/distro is blank
			## Allele is typical, didn't assign values in __atypical.py
			## Hence, set these values here (From seqpair object, where they reside)
			if not allele_object.get_rvdist():
				allele_object.set_fwassembly(self.sequencepair_object.get_fwassembly())
				allele_object.set_fwdist(self.sequencepair_object.get_fwdist())

			###############################
			## Stage one -- CCG Zygosity ##
			##############################

			##
			## Allele read peak (dsp error)
			major = max(self.reverse_aggregate)
			minor = max(n for n in self.reverse_aggregate if n != major)
			for item in [major, minor]:
				if item == 0 or item is None:
					raise ValueError('Insignificant read count. Cannot genotype.')
			tertiary = max(n for n in self.reverse_aggregate if n != major and n != minor)
			majidx = int(np.where(self.reverse_aggregate == major)[0][0])
			minidx = int(np.where(self.reverse_aggregate == minor)[0][0])
			## Check discrete diff between majidx and minidx
			abs_ratio = self.reverse_aggregate[minidx] / self.reverse_aggregate[majidx]


			## check if minor is n-1/n+1 AND third peak is np.close(minor)
			skip_flag = False
			if abs(majidx - minidx) == 1:
				if np.isclose([minor], [tertiary], atol=minor * 0.80):
					skip_flag = True
					self.sequencepair_object.set_ccguncertainty(True)
			if minor == self.reverse_aggregate[allele_object.get_ccg() - 1]:
				skip_flag = True
				self.sequencepair_object.set_ccguncertainty(True)
			## skip the following block if so
			if not skip_flag:
				if abs_ratio < 0.05:
					pass
				else:
					for fod_peak in [majidx + 1, minidx + 1]:
						if allele_object.get_ccg() not in [majidx + 1, minidx]:
							if fod_peak not in [self.sequencepair_object.get_primaryallele().get_ccg(),
												self.sequencepair_object.get_secondaryallele().get_ccg()]:
								allele_object.set_fodoverwrite(True)
								allele_object.set_ccgval(int(fod_peak))
				##
				## Check SVM didn't fail...
				if 1 < abs(majidx - minidx) < 10:
					peak_count = 0
					if abs_ratio < 0.05:
						pass
					## otherwise, perhaps SVM was wrong and missed a peak
					else:
						for peak in [majidx, minidx]:
							pmo = self.reverse_aggregate[peak - 1];
							ppo = self.reverse_aggregate[peak + 1]
							pmo_ratio = pmo / self.reverse_aggregate[peak]
							ppo_ratio = ppo / self.reverse_aggregate[peak]
							## calc ratio of peak+1/-1, if within range, add peak
							if pmo_ratio and ppo_ratio < 0.2:
								if 0.0 not in [pmo_ratio, ppo_ratio]:
									peak_count += 1
						## hotfix SVM results based on peak detection
						if peak_count == 2 and not self.zygosity_state == 'HETERO':
							if self.zygosity_state == 'HOMO':
								self.sequencepair_object.set_svm_failure(False)
								pass
							else:
								self.zygosity_state = 'HETERO'
								self.sequencepair_object.set_svm_failure(True)

				##
				## Clean up distribution for erroneous peaks
				## In the case of atypical alleles, unexpected peaks may exist in aggregates
				if self.zygosity_state == 'HOMO':
					for i in range(0, len(self.reverse_aggregate)):
						if np.isclose([i], [allele_object.get_ctg() - 1], atol=2):
							if np.isclose([self.reverse_aggregate[i]],
										  [self.reverse_aggregate[allele_object.get_ctg() - 1]],
										  atol=((self.reverse_aggregate[allele_object.get_ctg() - 1]) / 100) * 45):
								removal = (self.reverse_aggregate[i] / 100) * 57.5
								if i != allele_object.get_ctg() - 1:
									self.reverse_aggregate[i] -= removal
				else:
					for i in range(0, len(self.reverse_aggregate)):
						if i != allele_object.get_ctg() - 1:
							removal = (self.reverse_aggregate[i] / 100) * 75
							self.reverse_aggregate[i] -= removal

				allele_object.set_rvarray(self.reverse_aggregate)

			#################################
			## Stage two -- CTG continuity ##
			#################################
			index_inspection_count = 0
			if self.zygosity_state == 'HETERO': index_inspection_count = 2
			if self.zygosity_state == 'HOMO': index_inspection_count = 1
			inspections = self.index_inspector(index_inspection_count)
			for inspect in inspections:
				if np.isclose(allele_object.get_ctg(), [inspect[1]+1], atol=1):
					allele_object.set_validation(True)
			ctg_expectant.append(allele_object.get_ctg())

		try:
			if not ctg_expectant[0] == ctg_expectant[1]:
				self.expected_zygstate = 'HETERO'
			if ctg_expectant[0] == ctg_expectant[1]:
				self.expected_zygstate = 'HOMO'
		except IndexError:
			raise Exception('CTG Prediction Failure.')

		##
		## Check both alleles passed validation
		if (self.sequencepair_object.get_primaryallele().get_validation()) and (
				self.sequencepair_object.get_secondaryallele().get_validation()):
			return True
		else:
			return False


	def determine_ctg(self):

		##
		## Constructs
		pass_gtp = True


		##########################
		## Heterozygous for CTG ##
		##########################
		if self.zygosity_state == 'HETERO' or self.zygosity_state == 'HOMO*' or self.zygosity_state == 'HOMO+':
			for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:
				target_distro = distribution_split['CTG{}'.format(allele.get_ctg())]
				if self.zygosity_state == 'HOMO+':
					for i in range(0, len(target_distro)):
						if i != allele.get_cag() - 1:
							removal = (target_distro[i] / 100) * 85
							target_distro[i] -= removal
				allele.set_ctgthreshold(0.50)
		#		fod_failstate, cag_indexes = self.peak_detection(allele, target_distro, 1, 'CAGHet')
		#		while fod_failstate:
		#			fod_failstate, cag_indexes = self.peak_detection(allele, target_distro, 1, 'CAGHet', fod_recall=True)
		#		allele.set_fodcag(cag_indexes)

		########################
		## Homozygous for CTG ##
		########################
		if self.zygosity_state == 'HOMO':
			##
			## Double check CTG matches.. be paranoid
			primary_ctg = self.sequencepair_object.get_primaryallele().get_ctg()
			secondary_ctg = self.sequencepair_object.get_secondaryallele().get_ctg()
			try:
				if not primary_ctg == secondary_ctg:
					target = max(self.ctg_sum)[0]
					self.sequencepair_object.get_primaryallele().set_ctgval(target)
					self.sequencepair_object.get_secondaryallele().set_ctgval(target)
			except ValueError:
				max_array = [0, 0]
				for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:
					total_reads = sum(distro_split['CTG{}'.format(allele.get_ctg())])

					if total_reads > max_array[1]:
						max_array[1] = total_reads
						max_array[0] = allele.get_ctg()
				self.sequencepair_object.get_primaryallele().set_ctgval(max_array[0])
				self.sequencepair_object.get_secondaryallele().set_ctgval(max_array[0])

			##
			## Get distance estimate between two peaks in our target CTG distribution
			## set threshold to use in peak calling algorithm
			estimated_distance = abs(self.sequencepair_object.get_secondaryallele().get_ctg() -
									 self.sequencepair_object.get_primaryallele().get_ctg())

			if estimated_distance > 5: distance_threshold = 2
			elif estimated_distance == 1: distance_threshold = 0
			else: distance_threshold = 1

			##
			## Process each allele, getting the specific CTG distribution
			for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:
				target_distro = distribution_split['CTG{}'.format(allele.get_ctg())]

				allele.set_ctgthreshold(0.50)
				ctg_indexes = self.peak_detection(allele, target_distro, distance_threshold, 'CTGHom')
		return pass_gtp

	def genotype_validation(self):

		##
		## Constructs
		pass_vld = True
		primary_allele = self.sequencepair_object.get_primaryallele()
		secondary_allele = self.sequencepair_object.get_secondaryallele()
		ctg_zygstate = self.zygosity_state

		##
		## Subfunctions
		def read_comparison(val1, val2):
			if np.isclose(val1, val2, atol=1):
				return val2
			else:
				return val1
		##
		## Primary Allele
		primary_dsp_ctg = primary_allele.get_ctg()

		##
		## Secondary Allele
		secondary_dsp_ctg = secondary_allele.get_ctg()

		##
		## Double check forencode peaks
		def dimension_checker(input_list):

			fod = input_list[0]
			dsp = input_list[1]
			allele = input_list[2]

			for i in range(0, len(fod)):
				if np.isclose([fod[i]], [dsp], atol=1.0):
					allele.set_fodcag(fod[i])

		for item in [[primary_fod_cag, primary_dsp_cag, primary_allele], [secondary_fod_cag, secondary_dsp_cag, secondary_allele]]:
			dimension_checker(item)
			primary_fod_cag = primary_allele.get_fodcag(); secondary_fod_cag = secondary_allele.get_fodcag()

		##
		## Check for potential homozygous haplotype/neighbouring peak
		if ctg_zygstate == 'HOMO' and np.isclose(primary_dsp_cag, secondary_dsp_cag, atol=1):
			primary_target = distribution_split['CTG{}'.format(primary_allele.get_ctg())]
			secondary_target = distribution_split['CTG{}'.format(secondary_allele.get_ctg())]
			primary_reads = primary_target[primary_allele.get_cag()-1]
			secondary_reads = secondary_target[secondary_allele.get_cag()-1]
			diff = abs(primary_reads-secondary_reads)
			pcnt = (diff/max([primary_reads, secondary_reads]))
			##
			## If read count is so close (and distance is atol=1)
			## Neighbouring peak...
			if 0 < pcnt < 20:
				self.sequencepair_object.set_neighbouringpeaks(True)
				pass_vld = ensure_integrity()
				return pass_vld
			elif np.isclose([pcnt], [0.25], atol=0.05):
				self.sequencepair_object.set_neighbouringpeaks(True)
				pass_vld = ensure_integrity()
				return pass_vld
			elif primary_fod_cag.all() and secondary_fod_cag.all():
				self.sequencepair_object.set_homozygoushaplotype(True)
				self.sequencepair_object.set_secondary_allele(self.sequencepair_object.get_primaryallele())
				for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:
					if allele.get_peakreads() < 250:
						allele.set_fatalalignmentwarning(True)
						self.sequencepair_object.set_fatalreadallele(False)
					else:
						allele.set_fatalalignmentwarning(False)
						self.sequencepair_object.set_fatalreadallele(False)
				pass_vld = ensure_integrity()
				return pass_vld

		##
		## Check for diminished peaks (incase DSP failure / read count is low)
		## Primary read info
		primary_target = primary_dist['CTG{}'.format(primary_allele.get_ctg())]
		primary_reads = primary_target[primary_allele.get_cag() - 1]
		primary_total = sum(primary_target)
		## Secondary read info
		secondary_target = secondary_dist['CTG{}'.format(secondary_allele.get_ctg())]
		secondary_reads = secondary_target[secondary_allele.get_cag() - 1]
		secondary_total = sum(secondary_target)
		## Set specifics for zygstate
		peak_total = sum([primary_reads, secondary_reads]); dist_total = 0
		if ctg_zygstate == 'HOMO':
			dist_total = sum([primary_total])
		if ctg_zygstate == 'HOMO*' or ctg_zygstate == 'HOMO+':
			dist_total = sum([primary_total, secondary_total])
		if not peak_total/dist_total >= 0.65:
			if primary_fod_ctg == secondary_fod_ctg and primary_dsp_cag != secondary_dsp_cag:
				primary_target = distribution_split['CTG{}'.format(primary_allele.get_ctg())]
				split_target = primary_target[primary_allele.get_cag()+5:-1]
				difference_buffer = len(primary_target)-len(split_target)
				fod_failstate, cag_diminished = self.peak_detection(primary_allele, split_target, 1, 'CAGDim')
				if split_target[cag_diminished] > 100:
					if not primary_allele.get_allelestatus()=='Atypical' and not secondary_allele.get_allelestatus()=='Atypical':
						## bypass integrity checks
						secondary_allele.set_cagval(int(cag_diminished+difference_buffer-1))
						for peak in [primary_reads, secondary_reads]:
							if peak < 750:
								self.sequencepair_object.set_diminishedpeaks(True)
						return pass_vld

		return pass_vld

	def inspect_peaks(self):

		for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:

			target = distribution_split['CTG{}'.format(allele.get_ctg())]
			linspace = np.linspace(0,199,200)

			if allele.get_peakreads() < 250:
				allele.set_fatalalignmentwarning(True)
				self.sequencepair_object.set_fatalreadallele(True)

			##
			## fucking weird interp bug filtering
			## Interp a gaussian to suspected peak
			warnings.filterwarnings('error')
			try:
				warnings.warn(Warning())
				peaks_interp = peakutils.interpolate(linspace, target, ind=[allele.get_fodcag() - 1])
				if np.isclose([peaks_interp], [allele.get_fodcag() - 1], rtol=0.5):
					interp_distance = abs(peaks_interp - float(allele.get_fodcag()) - 1)
					allele.set_interpdistance(interp_distance[0])
				else:
					allele.raise_interpolation_warning(True)
			except Warning:
				allele.raise_interpolation_warning(True)
				pass

			##
			## Calculate % of reads located near peak
			spread_reads = sum(target[allele.get_cag()-6:allele.get_cag()+5])
			spread_pcnt = (spread_reads/sum(target))
			allele.set_vicinityreads(spread_pcnt)

			##
			## Calculate peak dropoff
			nminus = target[allele.get_cag()-2]; n = target[allele.get_cag()-1]; nplus = target[allele.get_cag()]
			nminus_overn = nminus/n; nplus_overn = nplus/n
			dropoff_list = [nminus_overn, nplus_overn]
			allele.set_immediate_dropoff(dropoff_list)

			##
			## Sometimes, alignment parameters can result in invalid genotyping (i.e. 2 peaks when expecting 1)
			## Test for this, inform user..
			if self.zygosity_state == 'HETERO':
				major = max(target)
				majoridx = np.where(target == major)[0][0]
				minor = max(n for n in target if n != major)
				minoridx = np.where(target == minor)[0][0]
				thresh = (major/100)*55

				if abs(majoridx-minoridx) > 2:
					if np.isclose([major],[minor], atol=thresh):
						allele.set_unexpectedpeaks(True)
						self.pass_vld = False

			##
			## Slippage
			## Gather from N-2:N-1, sum and ratio:N
			nmt = allele.get_cag() - 3; nmo = allele.get_cag() - 1
			slip_ratio = (sum(target[nmt:nmo])) / target[allele.get_cag() - 1]
			allele.set_backwardsslippage(slip_ratio)

			rv_ratio = (target[allele.get_fodcag()-2]/target[allele.get_fodcag()-1])
			fw_ratio = (target[allele.get_fodcag()]/target[allele.get_fodcag()-1])
			if not self.sequencepair_object.get_homozygoushaplotype() and not self.sequencepair_object.get_neighbouringpeaks():
				if np.isclose([fw_ratio], [0.85], atol=0.075):
					if rv_ratio > 0.65:
						allele.set_fodcag(allele.get_fodcag()+1)
						allele.set_slippageoverwrite(True)
				if np.isclose([rv_ratio], [0.80], atol=0.150):
					if fw_ratio > 0.65:
						allele.set_fodcag(allele.get_fodcag()-1)
						allele.set_slippageoverwrite(True)
			##
			## If we're not homozygous or neighbouring, 'normal' peaks..
			## Check dropoffs are legitimate and 'clean'
			if not self.sequencepair_object.get_homozygoushaplotype() and not self.sequencepair_object.get_neighbouringpeaks():
				self.close_check(allele, nminus_overn, [0.25], 0.02, 1, state='minus') ## inform user
				self.close_check(allele, nplus_overn, [0.05], 0.02, 1, state='plus')   ## inform user
				self.close_check(allele, nminus_overn, [0.35], 0.04, 2, state='minus') ## warn user
				self.close_check(allele, nplus_overn, [0.15], 0.03, 2, state='plus')   ## warn user
				self.close_check(allele, nminus_overn, [0.45], 0.05, 3, state='minus') ## severe warning
				self.close_check(allele, nplus_overn, [0.27], 0.03, 3, state='plus')   ## severe warning
				self.close_check(allele, nminus_overn, [0.60], 0.05, 4, state='minus') ## extreme warning
				self.close_check(allele, nplus_overn, [0.37], 0.05, 4, state='plus')   ## extreme warning
				self.close_check(allele, nminus_overn, [0.75], 0.05, 5, state='minus') ## failure
				self.close_check(allele, nplus_overn, [0.65], 0.05, 5, state='plus')   ## failure
				if nminus_overn > 0.75: allele.set_nminuswarninglevel(6); self.pass_vld = False ## failure
				if nplus_overn > 0.65: allele.set_npluswarninglevel(6); self.pass_vld = False   ## failure
			else:
				allele.set_nminuswarninglevel(2)
				allele.set_npluswarninglevel(2)

			##
			## Somatic mosaicism
			## Gather from N+1:N+10, sum and ratio:N
			npo = allele.get_cag(); npt = allele.get_cag()+10
			somatic_ratio = (sum(target[npo:npt]))/target[allele.get_cag()-1]
			allele.set_somaticmosaicism(somatic_ratio)

			##
			## If we get here; alleles are valid
			allele.set_ccgvalid(True)
			allele.set_cagvalid(True)
			allele.set_genotypestatus(True)

			novel_caacag = allele.get_reflabel().split('_')[1]; novel_ccgcca = allele.get_reflabel().split('_')[2]
			allele.set_allelegenotype('{}_{}_{}'.format(novel_caacag,novel_ccgcca,allele.get_cct()))

			##
			## If failed, write intermediate data to report
			if not self.pass_vld:
				inspection_logfi = os.path.join(self.sequencepair_object.get_predictpath(),
												'{}{}'.format(allele.get_header(), 'PeakInspectionLog.txt'))
				inspection_str = '{}  {}\n{}: {}\n{}: {}\n' \
								 '{}: {}\n{}: {}\n{}: {}\n' \
								 '{}: {}\n{}: {}\n{}: {}\n' \
								 '{}: {}\n{}: {}\n'.format(
								 '>> Peak Inspection Failure','Intermediate results log',
								 'Investigating CCG', allele.get_ccg(),
								 'Interpolation warning', allele.get_interpolation_warning(),
								 'Interpolation distance', allele.get_interpdistance(),
								 'Reads (%) surrounding peak', allele.get_vicinityreads(),
								 'Peak dropoff', dropoff_list,
								 'NMinus ratio', nminus_overn,
								 'NMinus warning', allele.get_nminuswarninglevel(),
								 'NPlus ratio', nplus_overn,
								 'NPlus warning', allele.get_npluswarninglevel(),
								 'Unexpected Peaks', allele.get_unexpectedpeaks())
				with open(inspection_logfi,'w') as logfi:
					logfi.write(inspection_str)
					logfi.close()
		return self.pass_vld

	def render_graphs(self):

		##
		## Data for graph rendering (prevents frequent calls/messy code [[lol]])
		pri_fodctg = self.sequencepair_object.get_primaryallele().get_fodctg()-1
		sec_fodctg = self.sequencepair_object.get_secondaryallele().get_fodctg()-1
		pri_fodcag = self.sequencepair_object.get_primaryallele().get_fodcag()-1
		sec_fodcag = self.sequencepair_object.get_secondaryallele().get_fodcag()-1
		pri_rvarray = self.sequencepair_object.get_primaryallele().get_rvarray()
		sec_rvarray = self.sequencepair_object.get_secondaryallele().get_rvarray()
		pri_fwarray = self.sequencepair_object.get_primaryallele().get_fwarray()
		predpath = self.sequencepair_object.get_predictpath()

		def graph_subfunction(x, y, axis_labels, xticks, peak_index, predict_path, file_handle, prefix='', graph_type=None, neg_anchor=False):
			x = np.linspace(x[0],x[1],x[2])
			plt.figure(figsize=(10, 6)); plt.title(prefix+self.sequencepair_object.get_label())
			plt.xlabel(axis_labels[0]); plt.ylabel(axis_labels[1])
			if graph_type == 'bar':
				if neg_anchor: xtickslabel = xticks[2]
				else: xtickslabel = [str(i-1) for i in xticks[2]]
				ppl.bar(x, y, grid='y', annotate=True, xticklabels=xtickslabel)
				plt.xticks(size=8)
			else:
				plt.xticks(np.arange(xticks[0][0], xticks[0][1], xticks[0][2]), xticks[2])
				plt.xlim(xticks[1][0], xticks[1][1])
				pplot(x,y,peak_index)
			peak_index = [i+1 for i in peak_index]
			plt.legend(['Genotype: {}'.format(peak_index)])
			plt.savefig(os.path.join(predict_path, file_handle), format='pdf')
			plt.close()

		def pagemerge_subfunction(graph_list, prediction_path, cc=tg_val, header=None, hplus=False):

			##
			## Readers and pages
			line_reader = PyPDF2.PdfFileReader(open(graph_list[0], 'rb')); line_page = line_reader.getPage(0)
			bar_reader = PyPDF2.PdfFileReader(open(graph_list[1], 'rb')); bar_page = bar_reader.getPage(0)

			##
			## Create new page (double width), append bar and line pages side-by-side
			translated_page = PyPDF2.pdf.PageObject.createBlankPage(None, bar_page.mediaBox.getWidth()*2, bar_page.mediaBox.getHeight())
			translated_page.mergeScaledTranslatedPage(bar_page, 1, 720, 0)
			translated_page.mergePage(line_page)

			##
			## Write to one PDF
			if hplus: suffix = 'AtypicalHomozyg'
			else: suffix = ''
			if not header: output_path = os.path.join(prediction_path, 'CCG{}CAGDetection_{}.pdf'.format(ccg_val, suffix))
			else: output_path = os.path.join(prediction_path, 'IntroCCG.pdf')
			writer = PyPDF2.PdfFileWriter()
			writer.addPage(translated_page)
			with open(output_path, 'wb') as f:
				writer.write(f)

			##
			## Return CAG plot path
			return output_path

		##########################################
		## SAMPLE CARD FOR GENOTYPE INFORMATION ##
		##########################################
		sample_pdf_path = os.path.join(predpath, '{}{}'.format(self.sequencepair_object.get_label(),'.pdf'))
		c = canvas.Canvas(sample_pdf_path, pagesize=(720,432))
		header_string = '{}{}'.format('Sample header: ', self.sequencepair_object.get_label())
		primary_string = '{}(CAG{}, CTG{}) ({}; {})'.format('Primary: ', self.sequencepair_object.get_primaryallele().get_fodcag(),
												 self.sequencepair_object.get_primaryallele().get_fodccg(),
												 self.sequencepair_object.get_primaryallele().get_allelestatus(),
												 self.sequencepair_object.get_primaryallele().get_allelegenotype())
		secondary_string = '{}(CAG{}, CCTG{}) ({}; {})'.format('Secondary: ', self.sequencepair_object.get_secondaryallele().get_fodcag(),
												   self.sequencepair_object.get_secondaryallele().get_fodccg(),
												   self.sequencepair_object.get_secondaryallele().get_allelestatus(),
												   self.sequencepair_object.get_secondaryallele().get_allelegenotype())

		##########################################################
		## Create canvas for sample 'intro card'				##
		## Set font colour depending on subsample/invalid/valid ##
		## invalid == atypical allele, no realignment			##
		## valid == atypical allele, realigned					##
		##########################################################
		if self.sequencepair_object.get_subsampleflag():
			if self.sequencepair_object.get_subsampleflag() == '0.05**':
				pass
			elif self.sequencepair_object.get_automatic_DSPsubsample():
				pass
			elif float(self.sequencepair_object.get_subsampleflag()) >= 0.5:
				pass
			else:
				c.setFillColorRGB(75, 0, 130)
				c.drawCentredString(360, 25, '!! Genotype derived from significantly subsampled data !!')
		if self.invalid_data:
			c.setFillColorRGB(255, 0, 0)
			c.drawCentredString(250, 50, '!! Atypical alleles without re-alignment !!')
		if not self.invalid_data:
			c.setFillColorRGB(0, 0, 0)
		c.drawCentredString(360, 256, header_string)
		c.drawCentredString(360, 236, primary_string)
		c.drawCentredString(360, 216, secondary_string)
		c.save()

		###############################################
		## CTG heterozygous example					 ##
		## i.e. CCTG two peaks, one CAG dist per peak ##
		###############################################
		if self.zygosity_state == 'HETERO' or self.zygosity_state == 'HOMO*' or self.zygosity_state == 'HOMO+':

			##
			## Render CCG graph, append path to allele path list
			## Merge intro_ccg card with sample CCG graph
			## Append merged intro_ccg to heterozygous list
			hetero_graphs = []; ccg_peaks = [int(pri_fodccg),int(sec_fodccg)]
			concat = np.asarray([a + b for a, b in zip(pri_rvarray,sec_rvarray)])
			graph_subfunction([0, 21, 20], concat, ['CCG Value', 'Read Count'], ([1, 20, 1], [1, 20], range(1,21)),
							  ccg_peaks, predpath, 'CCGDetection.pdf', graph_type='bar', neg_anchor=True)
			intro_card = pagemerge_subfunction([sample_pdf_path, os.path.join(predpath, 'CCGDetection.pdf')],
													predpath, ccg_val=0, header=True)
			hetero_graphs.append(intro_card)
			plt.close()

			##
			## For each CCG allele in this heterozygous sample
			for allele in [self.sequencepair_object.get_primaryallele(),self.sequencepair_object.get_secondaryallele()]:

				##
				## Data for this allele (peak detection graph)
				temp_graphs = []
				target_distro = distribution_split['CCG{}'.format(allele.get_ccg())]
				if self.zygosity_state == 'HOMO+':
					for i in range(0, len(target_distro)):
						if i != allele.get_cag() - 1:
							removal = (target_distro[i] / 100) * 75
							target_distro[i] -= removal
				if allele.get_rewrittenccg():
					peak_filename = 'CCG{}-CAGDetection_atypical_ccgdiff.pdf'.format(allele.get_fodccg())
					peak_prefix = '(CCG{}**) '.format(allele.get_fodccg())
				elif allele.get_unrewrittenccg():
					peak_filename = 'CCG{}-CAGDetection_atypical_ccgsame.pdf'.format(allele.get_fodccg())
					peak_prefix = '(CCG{}++) '.format(allele.get_fodccg())
				else:
					peak_filename = 'CCG{}-CAGDetection.pdf'.format(allele.get_fodccg())
					peak_prefix = '(CCG{}) '.format(allele.get_fodccg())
				peak_graph_path = os.path.join(predpath, peak_filename)
				## Render the graph, append to list, close plot
				graph_subfunction([0, 199, 200], target_distro, ['CAG Value', 'Read Count'],
								  ([1, 200, 50], [1, 200], [0,50,100,150,200]), [np.int64(allele.get_fodcag() - 1)],
								  predpath, peak_filename, prefix=peak_prefix)
				temp_graphs.append(peak_graph_path); plt.close()

				##
				## Inspect the peak (subslice)
				slice_range = range(allele.get_fodcag()-4, allele.get_fodcag()+7)
				if allele.get_rewrittenccg():
					slice_filename = 'CCG{}-Peak_atypical_ccgdiff.pdf'.format(allele.get_fodccg())
					slice_prefix = '(CCG{}**) '.format(allele.get_ccg())
				elif allele.get_unrewrittenccg():
					slice_filename = 'CCG{}-Peak_atypical_ccgsame.pdf'.format(allele.get_fodccg())
					slice_prefix = '(CCG{}++) '.format(allele.get_ccg())
				else:
					slice_filename = 'CCG{}-Peak.pdf'.format(allele.get_fodccg())
					slice_prefix = '(CCG{}) ' .format(allele.get_ccg())
				sub = target_distro[np.int64(allele.get_fodcag()-6):np.int64(allele.get_fodcag()+5)]
				## Render the graph, append to list, close plot
				graph_subfunction([0,10,11], sub, ['CAG Value', 'Read Count'], ([1,11,1], [1,11], slice_range),
								  [np.int64(allele.get_fodcag()-1)], predpath,slice_filename, prefix=slice_prefix, graph_type='bar')
				temp_graphs.append(os.path.join(predpath,slice_filename)); plt.close()

				##
				## Merge 'allele sample' into one page
				ccg_val = allele.get_fodccg()
				if allele.get_unrewrittenccg(): hplus = True
				else: hplus = False
				merged_graph = pagemerge_subfunction(temp_graphs, predpath, ccg_val, hplus=hplus)
				hetero_graphs.append(merged_graph)

			self.sequencepair_object.get_primaryallele().set_allelegraphs(hetero_graphs)
			self.sequencepair_object.get_secondaryallele().set_allelegraphs(hetero_graphs)

		##############################################
		## CCG homozygous example					##
		## i.e. CCG one peak, one CAG dist per peak ##
		##############################################
		if self.zygosity_state == 'HOMO':

			##
			##Data for homozygous graph(s)
			homo_graphs = []
			page_graphs = []
			target_ccg = 'CCG{}'.format(self.sequencepair_object.get_primaryallele().get_ccg())
			## Peak data
			peak_filename = 'CCG{}-CAGDetection.pdf'.format(self.sequencepair_object.get_primaryallele().get_fodccg())
			peak_prefix = '(CCG{}) '.format(self.sequencepair_object.get_primaryallele().get_ccg())
			altpeak_filename = 'CCG{}-Peak.pdf'.format(self.sequencepair_object.get_primaryallele().get_fodccg())
			ccg_peaks = [int(pri_fodccg),int(sec_fodccg)]; cag_peaks = [int(pri_fodcag),int(sec_fodcag)]
			## Subslice data
			pri_cag = self.sequencepair_object.get_primaryallele().get_cag()
			sec_cag = self.sequencepair_object.get_secondaryallele().get_cag()
			upper = max([pri_cag, sec_cag])
			if self.sequencepair_object.get_homozygoushaplotype(): lower = upper
			else: lower = max(n for n in [pri_cag, sec_cag] if n != upper)
			sub = target_distro[lower-6:upper+5]
			slice_range = range(lower-4,upper+7)

			##
			## Render the graph, append to list, close plot
			## Merge intro_ccg card with sample CCG graph
			## Append merged intro_ccg to homozygous list, append line/bar peak to page list
			graph_subfunction([0, 21, 20], pri_rvarray, ['CCG Value', 'Read Count'], ([1, 20, 1], [1, 20], range(1,21)),
							  ccg_peaks, predpath, 'CCGDetection.pdf', graph_type='bar', neg_anchor=True); plt.close()
			graph_subfunction([0, 199, 200], target_distro, ['CAG Value', 'Read Count'],
							  ([1, 200, 50], [1, 200], [0,50,100,150,200]), cag_peaks, predpath,
							  peak_filename, prefix=peak_prefix); plt.close()
			graph_subfunction([0, len(sub)-1, len(sub)], sub, ['CAG Value', 'Read Count'],
							  ([1, len(sub), 1], [1, len(sub)], slice_range), cag_peaks, predpath, altpeak_filename,
							  prefix=peak_prefix, graph_type='bar'); plt.close()
			intro_card = pagemerge_subfunction([sample_pdf_path, os.path.join(predpath, 'CCGDetection.pdf')],
													predpath, ccg_val=0, header=True)
			homo_graphs.append(intro_card)
			page_graphs.append(os.path.join(predpath, peak_filename))
			page_graphs.append(os.path.join(predpath, altpeak_filename))

			##
			## Merge 'allele sample' into one page
			ccg_val = self.sequencepair_object.get_primaryallele().get_fodccg()
			merged_graph = pagemerge_subfunction(page_graphs, predpath, ccg_val)

			## Combine CCG and CAG graphs
			homo_graphs.append(merged_graph)
			self.sequencepair_object.get_primaryallele().set_allelegraphs(homo_graphs)
			self.sequencepair_object.get_secondaryallele().set_allelegraphs(homo_graphs)

		############################################
		## Merge graphs into a single summary PDF ##
		############################################

		##
		## Allele graphs
		## Ensure uniqueness of entries in primary/secondary (i.e. no duplicating CCG graph)
		primary_graphs = self.sequencepair_object.get_primaryallele().get_allelegraphs()[0]
		secondary_graphs = self.sequencepair_object.get_primaryallele().get_allelegraphs()[0]
		sample_graphs = primary_graphs + secondary_graphs
		def set_orderpreserve(seq, idfun=None):
			if idfun is None:
				def idfun(x): return x
			seen = {}
			result = []
			for item in seq:
				marker = idfun(item)
				if marker in seen: continue
				seen[marker] = 1
				result.append(item)
			return result
		uniques = set_orderpreserve(sample_graphs)

		##
		## Merge alleles together
		merger = PyPDF2.PdfFileMerger()
		for pdf in uniques:
			merger.append(pdf)
		merger.write(sample_pdf_path)
		merger.close()

		##
		## Remove individual plots
		clean_target = []
		for target_file in os.listdir(predpath):
			if target_file.endswith(".pdf"):
				clean_target.append(os.path.join(predpath, target_file))
		for rmpdf in clean_target:
			if not '{}{}'.format(self.sequencepair_object.get_label(),'.pdf') in rmpdf:
				os.remove(rmpdf)

	def calculate_score(self):

		##
		## For both alleles
		for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:
			allele_log_fi = os.path.join(self.sequencepair_object.get_predictpath(), '{}{}'.format(allele.get_header(), '_PenaltiesLog.txt'))
			with open(allele_log_fi, 'a') as penfi:
				penfi.write('{}, {}\n'.format('Flag/Warning','Score Penalty'))

				##
				## Start score high, deduct for questionable calls..
				allele_confidence = 100

				##
				## Sample based genotyping flags
				if self.sequencepair_object.get_recallcount() == 7: allele_confidence -= 25; penfi.write('{}, {}\n'.format('Recall Count','-25'))
				if 7 > self.sequencepair_object.get_recallcount() > 4: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Recall Count','-15'))
				if 4 > self.sequencepair_object.get_recallcount() > 0: allele_confidence -= 5; penfi.write('{}, {}\n'.format('Recall Count', '-5'))
				else: allele_confidence += 10; penfi.write('{}, {}\n'.format('Recall Count', '+10'))

				if self.sequencepair_object.get_homozygoushaplotype():
					allele_confidence -= 15; penfi.write('{}, {}\n'.format('Homozygous Haplotype','-15'))
				elif self.sequencepair_object.get_neighbouringpeaks():
					allele_confidence -= 25; penfi.write('{}, {}\n'.format('Neighbouring Peaks', '-25'))
				else: allele_confidence += 15; penfi.write('{}, {}\n'.format('Normal Data','+15'))

				if self.sequencepair_object.get_diminishedpeaks():
					allele_confidence -= 15; penfi.write('{}, {}\n'.format('Diminished Peaks','-15'))


				##
				## Allele based genotyping flags
				## Allele typical/atypical structure
				if allele.get_allelestatus() == 'Atypical':
					allele_confidence -= 5; penfi.write('{}, {}\n'.format('Atypical Allele','-5'))
					if np.isclose([float(allele.get_atypicalpcnt())],[50.00],atol=5.00):
						allele_confidence -= 20; penfi.write('{}, {}\n'.format('Atypical reads (50%)','-20'))
					if np.isclose([float(allele.get_atypicalpcnt())],[80.00],atol=20.00):
						allele_confidence += 15; penfi.write('{}, {}\n'.format('Atypical reads (80%>)','+15'))
				if allele.get_allelestatus() == 'Typical':
					allele_confidence += 5; penfi.write('{}, {}\n'.format('Typical Allele', '+5'))
					if np.isclose([float(allele.get_typicalpcnt())],[50.00],atol=5.00):
						allele_confidence -= 20; penfi.write('{}, {}\n'.format('Typical reads (50%)','-20'))
					if np.isclose([float(allele.get_typicalpcnt())],[80.00],atol=15.00):
						allele_confidence += 15; penfi.write('{}, {}\n'.format('Typical reads (80%>)','+15'))

				##
				## Total reads in sample..
				if allele.get_totalreads() > 10000:	allele_confidence += 10; penfi.write('{}, {}\n'.format('High total read count', '+10'))
				elif allele.get_totalreads() < 1000: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Low total read count', '-15'))
				else: allele_confidence += 5; penfi.write('{}, {}\n'.format('Normal total read count','+5'))

				##
				## Variance of distribution utilised
				if allele.get_vicinityreads()*100 > 85.00: allele_confidence += 5; penfi.write('{}, {}\n'.format('Reads near peak','+5'))
				elif 84.99 > allele.get_vicinityreads()*100 > 65.00: allele_confidence -= 10; penfi.write('{}, {}\n'.format('Reads near peak','-10'))
				elif 64.99 > allele.get_vicinityreads()*100 > 45.00: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Reads near peak','-15'))
				elif 44.99 > allele.get_vicinityreads()*100 > 00.00: allele_confidence -= 20; penfi.write('{}, {}\n'.format('Reads near peak','-20'))

				##
				## Backwards slippage ratio ([N-2:N-1]/N]
				if 0.00 < allele.get_backwardsslippage() < 0.10: allele_confidence += 10; penfi.write('{}, {}\n'.format('Backwards slippage','+10'))
				elif 0.10 < allele.get_backwardsslippage() < 0.25: allele_confidence += 5; penfi.write('{}, {}\n'.format('Backwards slippage','+5'))
				elif allele.get_backwardsslippage() > 0.25: allele_confidence -= 1; penfi.write('{}, {}\n'.format('Backwards slippage','-1'))
				elif allele.get_backwardsslippage() > 0.45: allele_confidence -= 10; penfi.write('{}, {}\n'.format('Backwards slippage','-10'))
				elif allele.get_backwardsslippage() > 0.65: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Backwards slippage','-15'))
				elif 0.65 < allele.get_backwardsslippage() < 1.00: allele_confidence -= 20; penfi.write('{}, {}\n'.format('Backwards slippage','-20'))
				if allele.get_slippageoverwrite(): allele_confidence -= 25; penfi.write('{}, {}\n'.format('Slippage overwrite','-25'))

				##
				## Somatic mosiacisim ratio ([N+1:N+10]/N]
				if 0.000 < allele.get_somaticmosaicism() < 0.010: allele_confidence += 10; penfi.write('{}, {}\n'.format('Somatic mosaicism','+10'))
				elif 0.010 < allele.get_somaticmosaicism() < 0.015: allele_confidence += 5; penfi.write('{}, {}\n'.format('Somatic mosaicism','+5'))
				elif allele.get_somaticmosaicism() > 0.015: allele_confidence -= 1; penfi.write('{}, {}\n'.format('Somatic mosaicism','-1'))
				elif allele.get_somaticmosaicism() > 0.025: allele_confidence -= 10; penfi.write('{}, {}\n'.format('Somatic mosaicism','-10'))
				elif allele.get_somaticmosaicism() > 0.035: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Somatic mosaicism','-15'))
				elif 0.035 < allele.get_somaticmosaicism() < 0.100: allele_confidence -= 20; penfi.write('{}, {}\n'.format('Somatic mosaicism','-20'))
				elif allele.get_somaticmosaicism() > 0.100: allele_confidence -= 30; penfi.write('{}, {}\n'.format('Somatic mosaicism','-30'))

				##
				## Peak calling thresholds
				for contig in [allele.get_ctgthreshold()]:
					if contig != 0.5:
						if 0.5 > contig > 0.3: allele_confidence -= 5; penfi.write('{}, {}\n'.format('Peak calling threshold','-5'))
					else: allele_confidence += 10; penfi.write('{}, {}\n'.format('Peak calling threshold','+10'))

				##
				## Peak dropoff warnings
				for peak_position_error in [allele.get_nminuswarninglevel(), allele.get_npluswarninglevel()]:
					if peak_position_error == 0: allele_confidence += 10; penfi.write('{}, {}\n'.format('Surrounding read ratio','+10'))
					elif peak_position_error == 1: allele_confidence -= 5; penfi.write('{}, {}\n'.format('Surrounding read ratio', '-5'))
					elif 2 >= peak_position_error > 1: allele_confidence -= 10; penfi.write('{}, {}\n'.format('Surrounding read ratio','-10'))
					elif peak_position_error >= 5: allele_confidence -= 25; penfi.write('{}, {}\n'.format('Surrounding read ratio','-25'))
					else: allele_confidence -= 15; penfi.write('{}, {}\n'.format('Surrounding read ratio','-15'))
				##
				## Mapping percentage
				for map_pcnt in [allele.get_fwalnpcnt(), allele.get_rvalnpcnt()]:
					if map_pcnt > 90: allele_confidence += 25; penfi.write('{}, {}\n'.format('Mapping percentage', '+25'))
					elif 85 < map_pcnt < 90: allele_confidence += 10; penfi.write('{}, {}\n'.format('Mapping percentage', '+10'))
					else: allele_confidence -= 10; penfi.write('{}, {}\n'.format('Mapping percentage', '-10'))

				##
				## Warning penalty.. if triggered, no confidence
				if self.warning_triggered: allele_confidence -= 20; penfi.write('{}, {}\n'.format('Peak Inspection warning triggered','-20'))
				if self.sequencepair_object.get_ccguncertainty(): allele_confidence -= 10; penfi.write('{}, {}\n'.format('CCG Uncertainty','-10'))
				if self.sequencepair_object.get_alignmentwarning(): allele_confidence -= 15; penfi.write('{}, {}\n'.format('Low read count alignment warning','-15'))
				if allele.get_fatalalignmentwarning(): allele_confidence -= 40; penfi.write('{}, {}\n'.format('Fatal low read count alignment warning','-40'))

				##
				## If reflabel CAG and FOD CAG differ.. no confidence
				label_split = allele.get_reflabel().split('_')[0]
				if allele.get_allelestatus() == 'Atypical':
					if not np.isclose([int(allele.get_fodcag())],[int(label_split)],atol=1):
						allele_confidence = 0; penfi.write('{}, {}\n'.format('Atypical DSP:FOD inconsistency','-100'))

				##
				## Determine score (max out at 100), return genotype
				capped_confidence = sorted([0, allele_confidence, 100])[1]
				allele.set_alleleconfidence(capped_confidence)
				penfi.write('{}, {}\n\n'.format('Final score', capped_confidence))
				penfi.close()

	def set_report(self):

		for allele in [self.sequencepair_object.get_primaryallele(), self.sequencepair_object.get_secondaryallele()]:

			##
			## Report path for this allele
			allele_filestring = '{}{}{}'.format(allele.get_header(),allele.get_allelestatus(), '_AlleleReport.txt')
			report_path = os.path.join(self.sequencepair_object.get_predictpath(), allele_filestring)
			allele.set_allelereport(report_path)
			report_string = '{}{}\n\n{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n\n' \
							'{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}\n\n' \
							'{}\n{}{}\n{}{}\n{}{}\n{}{}\n{}{}'.format(
							'Allele Report>> ', self.sequencepair_object.get_label(),
							'Summary Information>>',
							'Genotype: ', allele.get_allelegenotype(),
							'Subsampling %: ', self.sequencepair_object.get_subsampleflag(),
							'Confidence: ', allele.get_alleleconfidence(),
							'CTG Uncertain: ', self.sequencepair_object.get_ctguncertainty(),
							'Structure Status: ', allele.get_allelestatus(),
							'Typical Pcnt: ', allele.get_typicalpcnt(),
							'Total Reads: ', allele.get_totalreads(),
							'Flags>>',
							'Recall Count: ', self.sequencepair_object.get_recallcount(),
							'Homozygous Haplotype: ', self.sequencepair_object.get_homozygoushaplotype(),
							'Neighbouring Peaks: ', self.sequencepair_object.get_neighbouringpeaks(),
							'Diminished Peaks: ', self.sequencepair_object.get_diminishedpeaks(),
							'Backwards Slippage: ', allele.get_backwardsslippage(),
							'Somatic Mosaicism: ', allele.get_somaticmosaicism(),
							'Slippage Overwritten: ', allele.get_slippageoverwrite(),
							'Peak Interpolation Warning: ', allele.get_interpolation_warning(),
							'Peak Interpolation Distance: ', allele.get_interpdistance(),
							'Low read-count alignment: ', self.sequencepair_object.get_alignmentwarning(),
							'Fatal low read-count: ', allele.get_fatalalignmentwarning(),
							'Data Quality>>',
							'Reads (%) surrounding peak: ', allele.get_vicinityreads(),
							'Immediate Dropoffs: ', allele.get_immediate_dropoff(),
							'N-1 Warning Level: ', allele.get_nminuswarninglevel(),
							'N+1 Warning Level: ', allele.get_npluswarninglevel(),
							'CTG Threshold: ', allele.get_ctgthreshold()
							)
			##
			## Write to file
			with open(report_path, 'w') as outfi:
				outfi.write(report_string)
				outfi.close()

	def get_report(self):

		self.allele_report = [self.sequencepair_object.get_primaryallele().get_allelereport(),
							  self.sequencepair_object.get_secondaryallele().get_allelereport()]
		return self.allele_report
