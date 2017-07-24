import pysam

##Imports
import os
import re
import regex
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

class DeterminePeaks:
    def __init__(self, sequencepair_object, instance_params):

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
        ## Turn results into objects
        primary_object = IndividualAllele();
        secondary_object = IndividualAllele()
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
            obj.set_atypicalpcnt(dat.get('AtypicalPcnt'))
            obj.set_fiveprime(dat.get('5PFlank'))
            obj.set_cagval(dat.get('EstimatedCAG'))
            obj.set_caacagval(dat.get('EstimatedCAACAG'))
            obj.set_ccgccaval(dat.get('EstimatedCCGCCA'))
            obj.set_ccgval(dat.get('EstimatedCCG'))
            obj.set_threeprime(dat.get('3PFlank'))
            obj.set_rewrittenccg(dat.get('RewrittenCCG'))
            obj.set_unrewrittenccg(dat.get('UnrewrittenCCG'))
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
        for read in assembly_targets:
            if read > read + 1:



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

            ##
            ## Counts of atypical/typical reads
            typical_count = 0; atypical_count = 0; intervening_population = []; fp_flanks = []; tp_flanks = [] ref_cag = []; ref_ccg = []; ref_cct = []

            if investigation[1] < 100:
                self.sequencepair_object.set_fatalreadallele(True)
                raise Exception('<100 aligned reads. Data un-usable.')

                ##
                ##
                ## For every read in this reference, get the aligned sequence
                ## Split into triplet sliding window list, remove any triplets that are < 3
                for read in self.assembly_object.fetch(reference=investigation[0]):
                    target_sequence = read.query_alignment_sequence
                    sequence_windows = [target_sequence[i:i + 3] for i in range(0, len(target_sequence), 3)]
                    sequence_windows = [x for x in sequence_windows if len(x) == 3]

                    ##
                    ## Append results to reference label
                    self.atypical_info[investigation[0]] = reference_dictionary

                if not self.sequencepair_object.get_broadflag():
                    os.remove(self.subsample_assembly)
                    os.remove(self.subsample_index)

    def organise_atypicals(self):

        ##
        ## Constructs
        sorted_info = sorted(self.atypical_info.iteritems(), key=lambda (x, y): y['TotalReads'],
                                             reverse=True)
        if len(sorted_info) != 3: raise IndexError('< 3 references in sorted top; alignment failure?')

        #
        ## Check % dropoff in read count between #2 and #3
        alpha_diff = float(abs(sorted_info[0][1]['TotalReads'] - sorted_info[1][1]['TotalReads']))
        beta_diff = float(abs(sorted_info[0][1]['TotalReads'] - sorted_info[2][1]['TotalReads']))
        sub_diff = float(abs(sorted_info[1][1]['TotalReads'] - sorted_info[2][1]['TotalReads']))
        alpha_drop = float(alpha_diff / sorted_info[0][1]['TotalReads'])
        beta_drop = float(beta_diff / sorted_info[0][1]['TotalReads'])
        sub_drop = float(sub_diff / sorted_info[1][1]['TotalReads'])

        ## Top1 always used
        primary_allele = sorted_info[0][1];
        primary_allele['Reference'] = sorted_info[0][0]
        secondary_allele = None

        ##
        ## CTG matches between #2/#3, potential peak skew
        ##TODO lmao this is fucking horrible
        ##TODO refactor this please            ##
            ## check #2 and #3
            for val in [sorted_info[1], sorted_info[2]]:
                top1_reads = primary_allele['TotalReads'];
                curr_reads = val[1].get('TotalReads')
                read_drop = abs(top1_reads - curr_reads) / top1_reads
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





                    # samfile = pysam.AlignmentFile("ex1.bam", "rb")

    #pairedreads = pysam.AlignmentFile("allpaired.bam", "wb", template=samfile)
    #for read in samfile.fetch():
    #    if read.is_paired:
    #        paireads.write(read)
    #return pairedreads


   # highest_lowest = read.reverse()

