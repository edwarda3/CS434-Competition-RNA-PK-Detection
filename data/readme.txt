Fasta file format

A sequence in FASTA format begins with a single-line description,
followed by the sequence. An RNA sequence consists of 4 nucleotides
A,C,G, and U.


Example
>id
ACGGGUACCGAAUUUAA


There are 3 possible base pairing in RNA secondary structures: 1)C
and pair with G (G also can pair with C) 2)A can pair with U (U can
pair with A) 3)G can pair with U (U can pair with G)

The C:G {or G:C} pairs are the most stable base pairing. Then A:U
{or U:A} and finally G:U {or U:G}




Feature file format.

The txt data files are tab separated. The first line of the file
contains a header specifying the name for each field. The remainder
of the file contains one example per line. The first column of the
file contains the ID of the sequence (consistent with the ones in
the FASTA format). The second column stores the labels, 0 for PK
free sequence and 1 for PK-present sequence.
