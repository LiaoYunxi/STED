import pandas as pd
import os
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