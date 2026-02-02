library(GenomicRanges)
library(dplyr)
library(ggplot2)
library(ggrepel)
library(IRanges)
# =================function=================
calculate_cutoff <- function(inputVector, drawPlot=TRUE,...){
  inputVector <- sort(inputVector)
  inputVector[inputVector<0]<-0 #set those regions with more control than ranking equal to zero
  slope <- (max(inputVector)-min(inputVector))/length(inputVector) #This is the slope of the line we want to slide. This is the diagonal.
  xPt <- floor(optimize(numPts_below_line,lower=1,upper=length(inputVector),myVector= inputVector,slope=slope)$minimum) #Find the x-axis point where a line passing through that point has the minimum number of points below it. (ie. tangent)
  y_cutoff <- inputVector[xPt] #The y-value at this x point. This is our cutoff.
  
  if(drawPlot){  #if TRUE, draw the plot
    plot(1:length(inputVector), inputVector,type="l",...)
    b <- y_cutoff-(slope* xPt)
    abline(v= xPt,h= y_cutoff,lty=2,col=8)
    points(xPt,y_cutoff,pch=16,cex=0.9,col=2)
    abline(coef=c(b,slope),col=2)
    title(paste("x=",xPt,"\ny=",signif(y_cutoff,3),"\nFold over Median=",signif(y_cutoff/median(inputVector),3),"x\nFold over Mean=",signif(y_cutoff/mean(inputVector),3),"x",sep=""))
    axis(1,sum(inputVector==0),sum(inputVector==0),col.axis="pink",col="pink") #Number of regions with zero signal
  }
  return(list(absolute=y_cutoff,overMedian=y_cutoff/median(inputVector),overMean=y_cutoff/mean(inputVector)))
}

#this is an accessory function, that determines the number of points below a diagnoal passing through [x,yPt]
numPts_below_line <- function(myVector,slope,x){
  yPt <- myVector[x]
  b <- yPt-(slope*x)
  xPts <- 1:length(myVector)
  return(sum(myVector<=(xPts*slope+b)))
}

convert_stitched_to_bed <- function(inputStitched, trackName, trackDescription, outputFile, splitSuper=TRUE, score=c(), superRows=c(), baseColor="0,0,0", superColor="255,0,0"){
  # --- 核心修改：GRanges 转换与 ID 生成 ---
  if (inherits(inputStitched, "GRanges")) {
    # 1. 转换为数据框
    inputStitched <- data.frame(
      CHROM = as.character(seqnames(inputStitched)),
      START = start(inputStitched),
      STOP  = end(inputStitched),
      stringsAsFactors = FALSE
    )
    
    # 2. 按照您的要求生成 REGION_ID: chr_start_end
    inputStitched$REGION_ID <- paste(inputStitched$CHROM, inputStitched$START, inputStitched$STOP, sep="_")
  }
  # ----------------------------------------
  
  # 检查数据是否为空
  if (nrow(inputStitched) == 0) {
    warning("输入数据为空 (0 rows)，跳过生成 BED 文件。")
    return(NULL)
  }
  
  # 计算矩阵列数
  n_cols <- 4 + ifelse(length(score) == nrow(inputStitched), 1, 0)
  outMatrix <- matrix(data="", ncol=n_cols, nrow=nrow(inputStitched))
  
  # 填充矩阵
  outMatrix[,1] <- as.character(inputStitched$CHROM)
  outMatrix[,2] <- as.character(inputStitched$START)
  outMatrix[,3] <- as.character(inputStitched$STOP)
  outMatrix[,4] <- as.character(inputStitched$REGION_ID) # 这里写入刚才生成的 chr_start_end
  
  # 如果有分数(score)，处理排名
  if(length(score) == nrow(inputStitched)){
    score <- rank(score, ties.method="first")
    score <- length(score) - score + 1 
    outMatrix[,5] <- as.character(score)
  }
  
  # 处理 Track Header 信息
  trackDescription <- paste(trackDescription, "\nCreated on ", format(Sys.time(), "%b %d %Y"), collapse="", sep="")
  trackDescription <- gsub("\n", "\t", trackDescription)
  tName <- gsub(" ", "_", trackName)
  
  # 写入文件
  cat('track name="', tName, '" description="', trackDescription, '" itemRGB=On color=', baseColor, "\n", sep="", file=outputFile)
  write.table(file=outputFile, outMatrix, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE, append=TRUE)
  
  # 处理 Super Enhancers
  if(splitSuper == TRUE){
    if(length(superRows) > 0){
      cat("\ntrack name=\"Super_", tName, '" description="Super ', trackDescription, '" itemRGB=On color=', superColor, "\n", sep="", file=outputFile, append=TRUE)
      # 使用 drop=FALSE 防止单行数据导致矩阵结构丢失
      write.table(file=outputFile, outMatrix[superRows, , drop=FALSE], sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE, append=TRUE)
    }
  }
}

convert_stitched_to_gateway_bed <- function(inputStitched, outputFileRoot, splitSuper=TRUE, score=c(), superRows=c()){
  
  # --- 1. 兼容性处理：如果是 GRanges，转为 Data Frame 并生成 ID ---
  if (inherits(inputStitched, "GRanges")) {
    inputStitched <- data.frame(
      CHROM = as.character(seqnames(inputStitched)),
      START = start(inputStitched),
      STOP  = end(inputStitched),
      stringsAsFactors = FALSE
    )
    # 生成格式: chr_start_end
    inputStitched$REGION_ID <- paste(inputStitched$CHROM, inputStitched$START, inputStitched$STOP, sep="_")
  }
  # -----------------------------------------------------------
  
  # 安全检查：如果数据为空，直接返回
  if (nrow(inputStitched) == 0) {
    warning("输入数据为空 (0 rows)，跳过生成 Gateway BED 文件。")
    return(NULL)
  }
  
  # 初始化 6 列矩阵 (Gateway BED 标准通常需要 6 列，第6列是 strand)
  outMatrix <- matrix(data="", ncol=6, nrow=nrow(inputStitched))
  
  outMatrix[,1] <- as.character(inputStitched$CHROM)
  outMatrix[,2] <- as.character(inputStitched$START)
  outMatrix[,3] <- as.character(inputStitched$STOP)
  outMatrix[,4] <- as.character(inputStitched$REGION_ID)
  
  # 处理分数 (Score)
  if(length(score) == nrow(inputStitched)){
    score <- rank(score, ties.method="first")
    score <- length(score) - score + 1  # 反转排名 (1是最高的)
    outMatrix[,5] <- as.character(score)
  } else {
    outMatrix[,5] <- "0" # 如果没有 score，填 0 占位
  }
  
  # 第 6 列：Strand (链方向)，默认为 '.'
  outMatrix[,6] <- "."
  
  # --- 写入文件 1: 所有 Enhancers ---
  outputFile1 = paste(outputFileRoot, '_Gateway_Enhancers.bed', sep='')
  # 注意：Gateway 格式通常不带 header，这里直接追加写入
  # 建议：如果是新文件，最好先清空或不使用 append=TRUE，为了保持原函数逻辑，这里保留 append=TRUE
  write.table(file=outputFile1, outMatrix, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE, append=TRUE)
  
  # --- 写入文件 2: Super Enhancers (如果需要) ---
  if(splitSuper == TRUE){
    outputFile2 = paste(outputFileRoot, '_Gateway_SuperEnhancers.bed', sep='')
    
    if(length(superRows) > 0){
      # 使用 drop=FALSE 防止单行数据变成向量导致报错
      write.table(file=outputFile2, outMatrix[superRows, , drop=FALSE], sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE, append=TRUE)
    }
  }
}

parse_gr_from_df <- function(df) {
  rn <- rownames(df)
  
  # 移除空行名或 NA 行名
  valid_idx <- !is.na(rn) & nzchar(rn)
  rn <- rn[valid_idx]
  
  # 拆分行名，只保留恰好有3部分的（chr_start_end）
  parts <- strsplit(rn, "_")
  len_ok <- lengths(parts) == 3
  parts <- parts[len_ok]
  rn_valid <- rn[len_ok]
  
  if (length(parts) == 0) {
    return(GRanges())  # 无有效行
  }
  
  chr <- sapply(parts, `[`, 1)
  start_str <- sapply(parts, `[`, 2)
  end_str <- sapply(parts, `[`, 3)
  
  # 尝试转换为整数，无法转换的变为 NA
  start <- as.integer(start_str)
  end <- as.integer(end_str)
  
  # 过滤掉任何含 NA 或 start > end 的区间
  keep <- !is.na(start) & !is.na(end) & (start <= end) & (start > 0) & (end > 0)
  
  if (!any(keep)) {
    return(GRanges())
  }
  
  GRanges(
    seqnames = chr[keep],
    ranges = IRanges(start = start[keep], end = end[keep]),
    rowname = rn_valid[keep]  # 可选：保留原始行名
  )
}

writeSuperEnhancer_table <- function(superEnhancer,description,outputFile,additionalData=NA){
  description <- paste("#",description,"\nCreated on ",format(Sys.time(), "%b %d %Y"),collapse="",sep="")
  description <- gsub("\n","\n#",description)
  cat(description,"\n",file=outputFile)
  if(is.matrix(additionalData)){
    if(nrow(additionalData)!=nrow(superEnhancer)){
      warning("Additional data does not have the same number of rows as the number of super enhancers.\n--->>> ADDITIONAL DATA NOT INCLUDED <<<---\n")
    }else{
      superEnhancer <- cbind(superEnhancer,additionalData)
      superEnhancer = superEnhancer[order(superEnhancer$enhancerRank),]
      
    }
  }
  print(outputFile)
  write.table(file=outputFile,superEnhancer,sep="\t",quote=FALSE,row.names=FALSE,append=TRUE)
}
# =================Configure paths=================

# rankBy_factor_base <- "Tumor" #prefix1
# enhancerName <- "topic19_seed2" #prefix2
# peak_dir ="/data/lyx/supplement/CRC/STED/CorEx_Tumor/model/"
# outFolder = '/data/lyx/supplement/CRC/SE/'

# Use：Rscript /data/lyx/supplement/ROSE_test/callSuperEnhancer.R "outFolder" "rankBy_factor_base" "enhancerName" "peak_dir"
args <- commandArgs(trailingOnly = TRUE)

print('THESE ARE THE ARGUMENTS')
print(args)
outFolder = args[1]
rankBy_factor_base = args[2]
enhancerName = args[3]
peak_dir =  args[4]
color2 = c('#e41a1c', "#F781BF")

if(!dir.exists(dirname(outFolder))) dir.create(dirname(outFolder), recursive = TRUE)

files <- list.files(
  path = peak_dir,
  pattern = "predicted_filtered_peaks\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

# =================1.Read data=================
results_list <- list()
for (file in files) {
  signal_file <- read.csv(file, row.names = "X")
  signal_file <- t(signal_file)  # 转置后：行=样本，列=特征
  cn <- unlist(strsplit(basename(file), "_"))[1]
  results_list[[cn]] <- signal_file
}

# 保存原始读取的数据备份
# saveRDS(results_list, file = paste0("./CorEx_",rankBy_factor_base,"/predicted_peak_signals.rds"))

# =================2.Define Stitched Regions (Super Enhancers)=================
gr_list <- lapply(results_list, parse_gr_from_df)

# 合并所有样本的 Peaks
all_peaks <- unlist(GRangesList(gr_list)) 
consensus_peaks <- reduce(all_peaks) # 基础合并

# SE 缝合：将距离小于 12.5kb 的区域合并
stitched_regions <- reduce(consensus_peaks, min.gapwidth = 12500)

cat("Stitched regions count:", length(stitched_regions), "\n")

# =================3.Construct the signal matrix=================
celltypes <- colnames(results_list[[1]])
sample_names <- names(results_list)

final_se_matrices <- list()

for(ct in celltypes){
  message("Processing cell type: ", ct)
  signal_matrix <- matrix(0, 
                          nrow = length(stitched_regions), 
                          ncol = length(sample_names))
  rownames(signal_matrix) <- as.character(stitched_regions) # 或者 paste(seqnames, start, end)
  colnames(signal_matrix) <- sample_names
  

  for(sample in sample_names){
    df <- results_list[[sample]]
    gr_sample <- gr_list[[sample]] 
    if(ct %in% colnames(df)){
      signal_vals <- df[, ct]
    } else {
      warning(paste("Cell type", ct, "not found in sample", sample))
      next
    }
    
    hits <- findOverlaps(gr_sample, stitched_regions)
    
    if (length(hits) > 0) {
      stitched_signals <- tapply(signal_vals[queryHits(hits)], 
                                 subjectHits(hits), 
                                 FUN = max)
      idx <- as.integer(names(stitched_signals))
      signal_matrix[idx, sample] <- stitched_signals
    }
  }
  final_se_matrices[[ct]] <- signal_matrix
}

df_list = list()
for (ct in celltypes) {
  signal_matrix <- final_se_matrices[[ct]]
  
  avg_signal <- rowMeans(signal_matrix, na.rm = TRUE)
  
  df_list[[ct]] <- avg_signal
}

se_average_signal_df <- as.data.frame(df_list)

print(head(se_average_signal_df))
dim(se_average_signal_df)
write.csv(se_average_signal_df,
          file = paste0("./CorEx_",rankBy_factor_base,"/predicted_celltype_average_peak_signals.csv"),
          quote = F)


region_names <- paste(
  seqnames(stitched_regions),
  start(stitched_regions),
  end(stitched_regions),
  sep = "_"
)
signal_matrix = se_average_signal_df
rownames(signal_matrix) = region_names
peak = data.frame(stitched_regions)
write.csv(peak,file = paste0("../SE/",rankBy_factor_base,"_STED_region.csv"),quote = F,row.names = F)

for(ct in colnames(signal_matrix)){
  current_rankBy_factor <- paste(rankBy_factor_base, ct, sep = ":")
  
  rankBy_vector = signal_matrix[,ct]
  rankBy_vector[rankBy_vector < 0] <- 0
  
  # FIGURING OUT THE CUTOFF
  cutoff_options <- calculate_cutoff(rankBy_vector, 
                                     drawPlot=FALSE,
                                     xlab=paste(current_rankBy_factor,'_enhancers'),
                                     ylab=paste(current_rankBy_factor,' Signal','- ',wceName), # wceName 需预先定义
                                     lwd=2,col=4)
  
  # These are the super-enhancers
  superEnhancerRows <- which(rankBy_vector > cutoff_options$absolute)
  typicalEnhancers = setdiff(1:nrow(signal_matrix), superEnhancerRows)
  
  enhancerDescription <- paste("Ranked by ", current_rankBy_factor, "\nUsing cutoff of ",
                               cutoff_options$absolute," for Super-Enhancers",sep="",collapse="")
  
  if (inherits(stitched_regions, "GRanges")) {
    stitched_regions_df <- data.frame(
      CHROM = as.character(seqnames(stitched_regions)),
      START = start(stitched_regions),
      STOP  = end(stitched_regions),
      stringsAsFactors = FALSE
    )
    stitched_regions_df$REGION_ID <- paste(stitched_regions_df$CHROM, stitched_regions_df$START, stitched_regions_df$STOP, sep="_")
  } else {
    stitched_regions_df <- stitched_regions
  }
  
  bedFileName = file.path(outFolder, paste(enhancerName, gsub(" ","_",ct), 'Enhancers_withSuper.bed', sep='_'))
  
  convert_stitched_to_bed(
    inputStitched = stitched_regions_df, 
    trackName = paste(current_rankBy_factor, "Enhancers"), 
    trackDescription = enhancerDescription, 
    outputFile = bedFileName, 
    score = rankBy_vector, 
    splitSuper = TRUE, 
    superRows = superEnhancerRows, 
    baseColor = "0,0,0", 
    superColor = "255,0,0" 
  )
  
  bedFileRoot = paste(outFolder, enhancerName, sep='')
  convert_stitched_to_gateway_bed(
    inputStitched = stitched_regions_df, 
    outputFileRoot = bedFileRoot, 
    splitSuper = TRUE, 
    score = rankBy_vector, 
    superRows = superEnhancerRows
  )
  
  true_super_enhancers <- stitched_regions_df[superEnhancerRows, , drop=FALSE]
  
  additionalTableData <- matrix(data=NA, ncol=2, nrow=nrow(stitched_regions_df))
  colnames(additionalTableData) <- c("enhancerRank", "isSuper")
  
  if(length(rankBy_vector) == nrow(stitched_regions_df)){
    additionalTableData[,1] <- nrow(stitched_regions_df) - rank(rankBy_vector, ties.method="first") + 1
  }
  
  additionalTableData[,2] <- 0
  if(length(superEnhancerRows) > 0){
    additionalTableData[superEnhancerRows, 2] <- 1
  }
  
  # Writing tables
  enhancerTableFile = file.path(outFolder, paste(enhancerName, gsub(" ","_",ct), 'AllEnhancers.table.txt', sep='_'))
  superTableFile = file.path(outFolder, paste(enhancerName, gsub(" ","_",ct), 'SuperEnhancers.table.txt', sep='_'))
  
  writeSuperEnhancer_table(stitched_regions_df, enhancerDescription, enhancerTableFile, additionalData= additionalTableData)
  writeSuperEnhancer_table(true_super_enhancers, enhancerDescription, superTableFile, additionalData= additionalTableData[superEnhancerRows,])
  
  # Analysis and Annotation
  se_table = read.table(enhancerTableFile, header = T)
  rownames(se_table) = se_table$REGION_ID
  
  rownames(stitched_regions_df) = stitched_regions_df$REGION_ID
  stitched_regions_df$rankBy_vector = rankBy_vector
  
  se_table$strenth = stitched_regions_df[rownames(se_table), "rankBy_vector"]
  se_table$Symbol = ""
  
  write.csv(se_table, file = file.path(outFolder, paste(rankBy_factor_base, gsub(" ","_",ct), 'STED_ann.csv', sep='_')),
            quote = F, row.names = F)
  

  if(TRUE){ 
    data <- se_table
    data$stitchedPeakRank = seq(1, nrow(data))
    data$label = ""
    
    point <- cutoff_options$absolute
    n <- length(superEnhancerRows)
    
    text <- paste('Cutoff:', round(point, 2), '\n', 'SE number:', n)
    data$strenth = as.numeric(data$strenth)
    
    p = ggplot(data[data$strenth > 0, ]) + 
      geom_point(aes(x = nrow(data) - stitchedPeakRank, 
                     y = strenth, 
                     color = as.factor(isSuper)), 
                 size = 1, show.legend = TRUE, alpha = 1) +
      labs(title = paste("CM SE -", ct)) + 
      xlab("Rank") + 
      ylab("Signal value") +
      theme_classic() + 
      theme(
        plot.title = element_text(size = 15, hjust = 0.5),
        axis.title = element_text(size = 15),
        axis.text = element_text(size = 12),
        legend.position = "none" 
      ) +
      ylim(0, max(data$strenth)) +
      scale_color_manual(values = color2) +
      geom_hline(yintercept = point, linetype = "dashed", size = 0.5, color = "grey") + 
      geom_vline(xintercept = nrow(data) - n, linetype = "dashed", size = 0.5, color = "grey") +
      annotate("text", x = nrow(data)*0.8, y = max(data$strenth)*0.8, label = text, parse = FALSE, size = 4) +
      geom_text_repel(data = subset(data, strenth > 0 & isSuper == 1 & label != ""), # 只有 label 不为空才显示
                      aes(x = nrow(data) - stitchedPeakRank, 
                          y = strenth, 
                          label = label),
                      size = 4, 
                      color = "black", 
                      max.overlaps = 20)
    
    plot_filename <- file.path(outFolder, paste(enhancerName, gsub(" ","_",ct), 'normal_SE_Plot.pdf', sep='_'))
    
    ggsave(filename = plot_filename, 
           plot = p, 
           width = 6, 
           height = 5, 
           dpi = 300)
  }
}