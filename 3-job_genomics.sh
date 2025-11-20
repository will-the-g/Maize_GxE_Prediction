#!/bin/bash

#SBATCH --job-name=genomics
## Commented out log file paths since HTCondor can't handle subdirectories
## #SBATCH --output=logs/job_genomics.txt
#SBATCH --output=job_genomics.txt
#SBATCH --partition=comp06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --time=02:00:00

set -e

## Commented out since we’re not making subdirectories
# mkdir -p output logs

## create a list of individuals to be used for VCF file
# Original: python3 -u src/create_individuals.py | tee "logs/individuals.txt"
python3 -u src/create_individuals.py > individuals.txt

# formats the csv properly for filtering
# Original: tail -n +2 output/individuals.csv | sed 's/^Hybrid//' > output/tmp.csv && mv output/tmp.csv output/individuals.csv
tail -n +2 individuals.csv | sed 's/^Hybrid//' > tmp.csv && mv tmp.csv individuals.csv

## filter VCF and create kinships matrices
vcftools --vcf "data/Training_Data/5_Genotype_Data_All_2014_2025_Hybrids.vcf" \
  --keep 'individuals.csv' \
  --recode --recode-INFO-all --out maize_indiv

# filters variants by Minor Allele Frequency (MAF)
vcftools --vcf maize_indiv.recode.vcf --maf 0.01 --recode --recode-INFO-all --out maize_maf001

# Removes SNPs in high linkage disequilibrium (LD)
plink --vcf maize_maf001.recode.vcf --double-id --indep-pairwise 100 20 0.9 --out maize_pruned

# creates new vcf file containing LD-pruned variants
plink --vcf maize_maf001.recode.vcf --double-id --extract maize_pruned.prune.in --recode vcf --out maize_pruned

# Runs kinship.R file
# Original: Rscript src/kinship.R > "logs/kinships.txt"
Rscript src/kinship.R > kinships.txt
