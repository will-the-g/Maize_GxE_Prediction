#!/bin/bash

#SBATCH --job-name=genomics
#SBATCH --output=logs/job_genomics.txt
#SBATCH --partition=comp06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=02:00:00

## configs
#module purge
#module load gcc/9.3.1 mkl/19.0.5 intel/19.0.5 R/4.2.2 vcftools/0.1.15 plink/5.2


set -e

mkdir -p output logs

## create a list of individuals to be used for VCF file
python3 -u src/create_individuals.py | tee "logs/individuals.txt"

# formats the csv properly for filtering
tail -n +2 output/individuals.csv | sed 's/^Hybrid//' > output/tmp.csv && mv output/tmp.csv output/individuals.csv

## filter VCF and create kinships matrices (you will need `vcftools` and `plink` here):
vcftools --vcf "data/Training Data/5_Genotype_Data_All_2014_2025_Hybrids.vcf" --keep 'output/individuals.csv' --recode --recode-INFO-all --out output/maize_indiv

# filters variants by Minor Allele Frequency (MAF) so any SNP where the MAF is less than 0.01 is removed
vcftools --vcf output/maize_indiv.recode.vcf --maf 0.01 --recode --recode-INFO-all --out output/maize_maf001
# Removes SNPs in high linkage disequilibrium (LD) keeping only quasi-independent ones
plink --vcf output/maize_maf001.recode.vcf --double-id --indep-pairwise 100 20 0.9 --out output/maize_pruned
# creates new vcf file containing LD-pruned variants
plink --vcf output/maize_maf001.recode.vcf --double-id --extract output/maize_pruned.prune.in --recode vcf --out output/maize_pruned
# Runs kinship.R file
Rscript src/kinship.R > "logs/kinships.txt"
