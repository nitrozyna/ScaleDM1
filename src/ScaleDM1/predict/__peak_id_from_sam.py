import pysam

def approx_peaks(AlignmentFile):
    samfile = pysam.AlignmentFile("ex1.bam", "rb")

    with open(samfile) as f:
        for read in samfile.fetch('chr1', 100, 120):

        print read






    # Remember to reverse the reads first from highest to lowest
    highest_lowest = read.reverse()