#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from argparse import ArgumentParser
from urllib.parse import quote

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API ------------------------------------------------------
from Pegasus.api import *


class MaizeGxEWorkflow:
    wf = None
    sc = None
    tc = None
    props = None

    dagfile = None
    wf_name = None
    wf_dir = None

    # --- Init ----------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_name = "maize-gxe"
        self.wf_dir = str(Path(__file__).parent.resolve())

    # --- Write files in directory --------------------------------------------
    def write(self):
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.tc.write()
        self.rc.write()
        self.wf.write()

    # --- Configuration (Pegasus Properties) ----------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()

        self.props["pegasus.integrity.checking"] = "none"
        return

    # --- Site Catalog --------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        local_storage_dir = os.path.join(self.wf_dir, "output")

        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, shared_scratch_dir).add_file_servers(
                FileServer("file://" + shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, local_storage_dir).add_file_servers(
                FileServer("file://" + local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name, arch=Arch.X86_64)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
            .add_pegasus_profiles(data_configuration="nonsharedfs")
            .add_pegasus_profiles(auxillary_local="true")
            .add_directories(
                Directory(
                    Directory.SHARED_SCRATCH, shared_scratch_dir
                ).add_file_servers(
                    FileServer("file://" + shared_scratch_dir, Operation.ALL)
                ),
                Directory(Directory.LOCAL_STORAGE, local_storage_dir).add_file_servers(
                    FileServer("file://" + local_storage_dir, Operation.ALL)
                ),
            )
        )
        self.sc.add_sites(local, exec_site)

    # --- Transformation Catalog (Executables and Containers) -----------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()
        maize_gxe = Container(
            "maize-gxe", Container.SINGULARITY, "docker://willtheg/maize:v1.2.4"
        )
        self.tc.add_containers(maize_gxe)

        transforms = {}
        for file in Path.cwd().glob("*.sh"):
            tc = Transformation(
                file.name,
                site="local",
                pfn=file.resolve(),
                is_stageable=True,
                container=maize_gxe,
            )
            transforms[file.name] = tc
            self.tc.add_transformations(tc)

        tc = Transformation(
            "src/blues.R",
            site="local",
            pfn=(file.parent / "src/blues.R").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["1-job_blues.sh"].add_requirement(tc)

        tc = Transformation(
            "src/create_datasets.py",
            site="local",
            pfn=(file.parent / "src/create_datasets.py").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["2-job_datasets.sh"].add_requirement(tc)

        tc = Transformation(
            "src/preprocessing.py",
            site="local",
            pfn=(file.parent / "src/preprocessing.py").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["2-job_datasets.sh"].add_requirement(tc)
        transforms["5-job_e.sh"].add_requirement(tc)

        tc = Transformation(
            "src/create_individuals.py",
            site="local",
            pfn=(file.parent / "src/create_individuals.py").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["3-job_genomics.sh"].add_requirement(tc)

        tc = Transformation(
            "src/kinship.R",
            site="local",
            pfn=(file.parent / "src/kinship.R").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["3-job_genomics.sh"].add_requirement(tc)

        tc = Transformation(
            "src/kronecker.R",
            site="local",
            pfn=(file.parent / "src/kronecker.R").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["4-job_kroneckers.sh"].add_requirement(tc)

        tc = Transformation(
            "src/run_e_model.py",
            site="local",
            pfn=(file.parent / "src/run_e_model.py").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["5-job_e.sh"].add_requirement(tc)
        tc = Transformation(
            "src/evaluate.py",
            site="local",
            pfn=(file.parent / "src/evaluate.py").resolve(),
            is_stageable=True,
        )
        self.tc.add_transformations(tc)
        transforms["5-job_e.sh"].add_requirement(tc)

    # --- Replica Catalog (Executables and Containers) ------------------------
    def create_replica_catalog(self, data_dir):
        self.rc = ReplicaCatalog()

        for file in Path(data_dir).iterdir():
            self.rc.add_replica(
                "local", f"data/Training_Data/{file.name}", file.resolve()
            )
        for file in (Path(data_dir).parent / "Testing_Data").iterdir():
            self.rc.add_replica(
                "local", f"data/Testing_Data/{file.name}", file.resolve()
            )

    # --- Create Workflow -----------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        job_blues = (
            Job("1-job_blues.sh")
            .add_inputs("data/Training_Data/1_Training_Trait_Data_2014_2021.csv")
            .add_outputs("logs/blues.txt", stage_out=True, register_replica=False)
            .add_outputs("blues.csv", stage_out=True, register_replica=False)
            .add_outputs("cvs_h2s.csv", stage_out=True, register_replica=False)
        )

        logs = []
        xtrain = set()
        xval = set()
        ytrain = set()
        yval = set()
        feat_imp_e_model_fold = set()
        oof_e_model_fold = set()
        pred_train_e_model_fold = set()
        logs_e_model_cv = set()
        for cv in range(3):
            for fold in range(5):
                for seed in range(1, 11):
                    xtrain.add(f"cv{cv}_xtrain_fold{fold}_seed{seed}.csv")
                    xval.add(f"cv{cv}_xval_fold{fold}_seed{seed}.csv")
                    ytrain.add(f"cv{cv}_ytrain_fold{fold}_seed{seed}.csv")
                    yval.add(f"cv{cv}_yval_fold{fold}_seed{seed}.csv")
                    logs.append(f"logs/datasets_cv${cv}_fold${fold}_seed${seed}.txt")

                    feat_imp_e_model_fold.add(
                        f"cv{cv}_feat_imp_e_model_fold{fold}_seed{seed}.csv"
                    )
                    oof_e_model_fold.add(
                        f"cv{cv}_oof_e_model_fold{fold}_seed{seed}.csv"
                    )
                    pred_train_e_model_fold.add(
                        f"cv{cv}_pred_train_e_model_fold{fold}_seed{seed}.csv"
                    )
                    logs_e_model_cv.add(
                        f"logs/e_model_cv{cv}_fold{fold}_seed{seed}.txt"
                    )

        job_datasets = (
            Job("2-job_datasets.sh")
            .add_inputs("blues.csv")
            .add_inputs("data/Training_Data/1_Training_Trait_Data_2014_2021.csv")
            .add_inputs("data/Training_Data/2_Training_Meta_Data_2014_2021.csv")
            .add_inputs("data/Training_Data/3_Training_Soil_Data_2015_2021.csv")
            .add_inputs("data/Training_Data/4_Training_Weather_Data_2014_2021.csv")
            .add_inputs("data/Training_Data/6_Training_EC_Data_2014_2021.csv")
            .add_inputs("data/Training_Data/All_hybrid_names_info.csv")
            .add_inputs("data/Testing_Data/1_Submission_Template_2022.csv")
            .add_inputs("data/Testing_Data/2_Testing_Meta_Data_2022.csv")
            .add_inputs("data/Testing_Data/3_Testing_Soil_Data_2022.csv")
            .add_inputs("data/Testing_Data/4_Testing_Weather_Data_2022.csv")
            .add_inputs("data/Testing_Data/6_Testing_EC_Data_2022.csv")
            .add_outputs(*xtrain, stage_out=True, register_replica=False)
            .add_outputs(*xval, stage_out=True, register_replica=False)
            .add_outputs(*ytrain, stage_out=True, register_replica=False)
            .add_outputs(*yval, stage_out=True, register_replica=False)
        )

        job_genomics = (
            Job("3-job_genomics.sh")
            .add_inputs(*ytrain)
            .add_inputs(*yval)
            .add_inputs("data/Training_Data/5_Genotype_Data_All_2014_2025_Hybrids.vcf")
            .add_outputs("logs/individuals.txt", stage_out=True, register_replica=False)
            .add_outputs(
                "individuals.csv", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_indiv.recode.vcf", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_maf001.recode.vcf", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_pruned.prune.in", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_pruned.prune.out", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_pruned.nosex", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_pruned.vcf", stage_out=True, register_replica=False
            )
            .add_outputs(
                "kinship_additive.txt", stage_out=True, register_replica=False
            )
            .add_outputs(
                "kinship_dominant.txt", stage_out=True, register_replica=False
            )
            .add_outputs(
                "maize_pruned.log", stage_out=True, register_replica=False
            )
            .add_outputs("logs/kinships.txt", stage_out=True, register_replica=False)
        )

        job_kroneckers = (
            Job("4-job_kroneckers.sh")
            .add_inputs(*xtrain)
            .add_inputs(*xval)
            .add_inputs(*ytrain)
            .add_inputs(*yval)
            .add_inputs(
                *[
                    f"kinship_{kinship}.txt"
                    for kinship in ("additive", "dominant")
                ]
            )
            .add_outputs(
                *[
                    f"logs/kronecker_{kinship}_cv{cv}.txt"
                    for cv in range(3)
                    for kinship in ("additive", "dominant")
                ],
                stage_out=True,
                register_replica=False,
            )
            .add_outputs(
                *[
                    f"cv{cv}_kronecker_{kinship}.arrow"
                    for cv in range(3)
                    for kinship in ("additive", "dominant")
                ],
                stage_out=True,
                register_replica=False,
            )
        )

        job_e = (
            Job("5-job_e.sh")
            .add_inputs(*xtrain)
            .add_inputs(*xval)
            .add_inputs(*ytrain)
            .add_inputs(*yval)
            .add_inputs(
                "data/Training_Data/1_Training_Trait_Data_2014_2021.csv",
                "data/Testing_Data/1_Submission_Template_2022.csv",
                "data/Training_Data/2_Training_Meta_Data_2014_2021.csv",
                "data/Testing_Data/2_Testing_Meta_Data_2022.csv",
            )
            .add_outputs(*logs_e_model_cv, stage_out=True, register_replica=False)
            .add_outputs(*feat_imp_e_model_fold, stage_out=True, register_replica=False)
            .add_outputs(*oof_e_model_fold, stage_out=True, register_replica=False)
            .add_outputs(
                *pred_train_e_model_fold, stage_out=True, register_replica=False
            )
        )

        self.wf.add_jobs(job_blues, job_datasets, job_genomics, job_kroneckers, job_e)


if __name__ == "__main__":
    parser = ArgumentParser(description="Pegasus Maize GxE Workflow")
    parser.add_argument(
        "-d",
        "--data",
        default="data/Training_Data/",
        help="Input data directory",
    )
    parser.add_argument(
        "-s",
        "--skip_sites_catalog",
        action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e",
        "--execution_site_name",
        metavar="STR",
        type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="STR",
        type=str,
        default="workflow.yml",
        help="Output file (default: workflow.yml)",
    )

    args = parser.parse_args()

    workflow = MaizeGxEWorkflow(args.output)

    if not args.skip_sites_catalog:
        print("Creating execution sites...")
        workflow.create_sites_catalog(args.execution_site_name)

    print("Creating workflow properties...")
    workflow.create_pegasus_properties()

    print("Creating transformation catalog...")
    workflow.create_transformation_catalog(args.execution_site_name)

    print("Creating replica catalog...")
    workflow.create_replica_catalog(args.data)

    print("Creating process workflow dag...")
    workflow.create_workflow()

    workflow.write()
