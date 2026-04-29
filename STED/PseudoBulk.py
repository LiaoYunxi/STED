import pandas as pd
import numpy as np
import os
import json
import pyranges as pr
import gzip
import anndata as ad
import pybedtools
import argparse

def featureCounts(bed_file,fragment_file,file_name='peaks_counts.txt',barcodes=None,chromosomes=None,sample_name=None,use_polars=True):
    print(f"Running featureCounts with:")
    print(f"  BED: {bed_file}")
    print(f"  Fragments: {fragment_file}")
    print(f"  Output: {file_name}")
    print(f"  Use Polars: {use_polars}")
    
    all_peak = pd.read_table(bed_file,sep='\t',header=None)
    if sample_name:
        fragments_dict = {sample_name:fragment_file}
    else:
        fragments_dict = {"pseudobulk":fragment_file} 
    fragments_df = read_fragments_from_file(fragments_dict['10x_pbmc'], use_polars=use_polars)

    if chromosomes is None:
        chromosomes = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']

    fragments_df = fragments_df[fragments_df.Chromosome.isin(chromosomes)]

    if barcodes:
        barcodes = set(barcodes)
        fragments_df = fragments_df[fragments_df.Name.isin(barcodes)]
    fragments_df = fragments_df.drop("Name")

    pbt_peaks = pybedtools.BedTool.from_dataframe(all_peak)
    pbt_fragments = pybedtools.BedTool.from_dataframe(fragments_df.df)
    pbt_overlap = pbt_peaks.intersect(pbt_fragments,wa=True,wb=True)
    pbt_merge = pbt_overlap.sort().merge(c=7, o=['sum'])
    df_overlap = pbt_merge.to_dataframe(names=['Chromosome','Start','End','Score'])
    df_overlap['peak']=[str(df_overlap['Chromosome'][i])+'_'+str(df_overlap['Start'][i])+'_'+
                    str(df_overlap['End'][i]) for i in range(0,df_overlap.shape[0])]
    df_overlap.to_csv(file_name,sep="\t")


def read_fragments_from_file(
    fragments_bed_filename, use_polars: bool = True
) -> pr.PyRanges:
    """
    Read fragments BED file to PyRanges object.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.
    use_polars: Use polars instead of pandas for reading the fragments BED file.

    Returns
    -------
    PyRanges object of fragments.
    """

    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0
    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if nbr_columns < 4:
        raise ValueError(
            f'Fragments BED file needs to have at least 4 columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    if use_polars:
        import polars as pl

        # Read fragments BED file with polars.
        df = (
            pl.read_csv(
                fragments_bed_filename,
                has_header=False,
                skip_rows=skip_rows,
                separator="\t",
                use_pyarrow=True,
                new_columns=bed_column_names[:nbr_columns],
            )
            .with_columns(
                [
                    pl.col("Chromosome").cast(pl.Utf8),
                    pl.col("Start").cast(pl.Int32),
                    pl.col("End").cast(pl.Int32),
                    pl.col("Name").cast(pl.Utf8),
                ]
            )
            .to_pandas()
        )

        # Convert "Name" column to pd.Categorical as groupby operations will be done on it later.
        df["Name"] = df["Name"].astype("category")
    else:
        # Read fragments BED file with pandas.
        df = pd.read_table(
            fragments_bed_filename,
            sep="\t",
            skiprows=skip_rows,
            header=None,
            names=bed_column_names[:nbr_columns],
            doublequote=False,
            engine="c",
            dtype={
                "Chromosome": str,
                "Start'": np.int32,
                "End": np.int32,
                "Name": "category",
                "Strand": str,
            },
        )

    # Convert pandas dataframe to PyRanges dataframe.
    # This will convert "Chromosome" and "Strand" columns to pd.Categorical.
    return pr.PyRanges(df)


def generate_pseudobulk_with_scenario(
    sc_adata_file: str,
    peak_file: str,
    out_dir: str,
    celltype_col: str = "cell_type",
    rare_cell_fraction: float = None,
    rare_celltype: str = None,
    signal_attenuation: float = 1.0,
    random_state: int = 42,
):
    """Generate pseudo-bulk profiles with controllable simulation scenarios.

    Extends the standard pseudo-bulk generation with support for rare-cell
    downsampling and signal attenuation, answering Reviewer 2 Major Comment 1
    question 4 about STED's performance under sparse/rare/weak-signal conditions.

    Parameters
    ----------
    sc_adata_file : str
        Path to h5ad file with single-cell ATAC-seq or multiome data.
    peak_file : str
        Path to h5ad file with peak counts.
    out_dir : str
        Output directory for pseudo-bulk files.
    celltype_col : str
        Column name in sc_adata.obs for cell type annotation.
    rare_cell_fraction : float, optional
        If set, downsample one cell type to this fraction of its original
        count before pseudo-bulk generation (e.g., 0.01 for 1%).
    rare_celltype : str, optional
        Which cell type to downsample. If None and rare_cell_fraction is set,
        the smallest cell type is chosen automatically.
    signal_attenuation : float
        Global multiplier for peak signal intensity (1.0 = no attenuation,
        0.5 = half signal, etc.).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Metadata about the simulation scenario.
    """
    np.random.seed(random_state)
    os.makedirs(out_dir, exist_ok=True)

    adata = ad.read_h5ad(sc_adata_file)
    peak_adata = ad.read_h5ad(peak_file)

    celltypes = adata.obs[celltype_col].unique().tolist()
    cell_counts = adata.obs[celltype_col].value_counts().to_dict()

    # Downsample rare cell type if requested
    if rare_cell_fraction is not None:
        if rare_celltype is None:
            rare_celltype = min(cell_counts, key=cell_counts.get)

        original_count = cell_counts[rare_celltype]
        target_count = max(1, int(original_count * rare_cell_fraction))

        # Get all cells of the rare type
        rare_cells = adata.obs_names[adata.obs[celltype_col] == rare_celltype]
        # Randomly subsample
        keep_cells = np.random.choice(rare_cells, size=target_count, replace=False)

        # Build filter mask
        mask = np.ones(adata.n_obs, dtype=bool)
        rare_idx = [adata.obs_names.to_list().index(c) for c in rare_cells]
        drop_idx = [i for i, c in enumerate(rare_cells) if c not in keep_cells]
        for i in drop_idx:
            mask[rare_idx[i]] = False

        adata = adata[mask].copy()
        cell_counts_after = adata.obs[celltype_col].value_counts().to_dict()
    else:
        cell_counts_after = cell_counts

    # Generate pseudo-bulk per cell type
    pseudobulk_profiles = {}
    for ct in celltypes:
        ct_cells = adata.obs_names[adata.obs[celltype_col] == ct]
        if len(ct_cells) == 0:
            continue
        ct_idx = [adata.obs_names.to_list().index(c) for c in ct_cells]
        ct_mat = adata[ct_cells].X
        if hasattr(ct_mat, "toarray"):
            ct_mat = ct_mat.toarray()
        pseudobulk_profiles[ct] = ct_mat.sum(axis=0).flatten()

    # Apply signal attenuation
    if signal_attenuation != 1.0:
        for ct in pseudobulk_profiles:
            pseudobulk_profiles[ct] = pseudobulk_profiles[ct] * signal_attenuation

    # Save pseudo-bulk
    pseudobulk_df = pd.DataFrame.from_dict(pseudobulk_profiles, orient="index")
    pseudobulk_df.columns = peak_adata.var_names if peak_adata.n_vars == pseudobulk_df.shape[1] else range(pseudobulk_df.shape[1])

    # Save ground truth proportions
    total_cells = sum(cell_counts_after.values())
    proportions = {ct: count / total_cells for ct, count in cell_counts_after.items()}

    # Save outputs
    pseudobulk_df.to_csv(os.path.join(out_dir, "pseudobulk.tsv"), sep="\t")
    pd.Series(proportions).to_csv(os.path.join(out_dir, "ground_truth_proportions.tsv"), sep="\t", header=["proportion"])

    # Save scenario metadata
    scenario_meta = {
        "rare_cell_fraction": rare_cell_fraction,
        "rare_celltype": rare_celltype,
        "signal_attenuation": signal_attenuation,
        "random_state": random_state,
        "original_cell_counts": {str(k): int(v) for k, v in cell_counts.items()},
        "modified_cell_counts": {str(k): int(v) for k, v in cell_counts_after.items()},
        "ground_truth_proportions": proportions,
    }

    with open(os.path.join(out_dir, "scenario_meta.json"), "w") as f:
        json.dump(scenario_meta, f, indent=2)

    return scenario_meta


if __name__ == "__main__":
    # 1. 创建解析器对象
    parser = argparse.ArgumentParser(
        description="FeatureCounts: Count fragments in peaks/regions."
    )

    # 2. 添加必须参数 (Positional Arguments)
    parser.add_argument("bed_file", type=str, help="Path to the BED file (e.g., peaks.bed)")
    parser.add_argument("fragment_file", type=str, help="Path to the fragment file (e.g., fragments.tsv.gz)")

    # 3. 添加可选参数 (Optional Arguments)
    parser.add_argument("-o", "--output", dest="file_name", default="peaks_counts.txt", 
                        help="Output file name (default: peaks_counts.txt)")
    
    parser.add_argument("--barcodes", type=str, default=None, 
                        help="Path to file containing selected barcodes (optional)")
    
    parser.add_argument("--chromosomes", nargs="+", default=None, 
                        help="Space-separated list of chromosomes to keep (e.g., chr1 chr2)")

    parser.add_argument("--sample-name", dest="sample_name", default=None, 
                        help="Sample name tag to add to output (optional)")

    # 4. 处理布尔值 use_polars (默认为 True，添加 flag 来关闭它)
    parser.add_argument("--no-polars", dest="use_polars", action="store_false", 
                        help="Disable Polars and use standard Pandas (slower but compatible)")
    # 显式设置默认值为 True (虽然函数默认也是 True，但这在 argparse 中是个好习惯)
    parser.set_defaults(use_polars=True)

    # 5. 解析参数
    args = parser.parse_args()

    # 6. (可选) 如果 barcodes 传入的是文件路径，但函数需要列表，可以在这里读取
    # if args.barcodes:
    #     with open(args.barcodes, 'r') as f:
    #         args.barcodes = [line.strip() for line in f]

    # 7. 调用函数
    featureCounts(
        bed_file=args.bed_file,
        fragment_file=args.fragment_file,
        file_name=args.file_name,
        barcodes=args.barcodes,
        chromosomes=args.chromosomes,
        sample_name=args.sample_name,
        use_polars=args.use_polars
    )