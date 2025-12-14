suppressPackageStartupMessages({
  library(optparse)
  library(msigdbr)
  library(gplots)
  library(dplyr)
  library(clusterProfiler)
  library(ggplot2)
  library(ggnewscale)
})

# --- 定义参数列表 ---
option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL, 
              help = "【必填】输入主目录路径 (包含 model 文件夹)", metavar = "PATH"),
  
  make_option(c("-g", "--genome"), type = "character", default = "human", 
              help = "参考基因组版本 [默认: %default]", metavar = "STR"),
  
  make_option(c("-s", "--seed"), type = "integer", default = 123, 
              help = "随机种子 [默认: %default]", metavar = "INT"),
  
  make_option(c("-t", "--threshold"), type = "integer", default = 100, 
              help = "Cluster筛选阈值 [默认: %default]", metavar = "INT"),
  
  make_option(c("-n", "--minGS"), type = "integer", default = 10, 
              help = "GO分析最小基因集大小 [默认: %default]", metavar = "INT"),
  
  make_option(c("-x", "--maxGS"), type = "integer", default = 500, 
              help = "GO分析最大基因集大小 [默认: %default]", metavar = "INT")
)

# --- 解析参数 ---
opt_parser <- OptionParser(option_list = option_list, description = "TopicAnnotation 分析脚本")
opt <- parse_args(opt_parser)

# --- 检查必填参数 ---
if (is.null(opt$input)) {
  print_help(opt_parser)
  stop("错误: 必须提供输入目录参数 (-i 或 --input)！", call. = FALSE)
}

# --- 赋值给原有变量 (保持后续代码逻辑不变) ---
base_dir  <- opt$input
genome    <- opt$genome
seed      <- opt$seed
threshold <- opt$threshold
minGSSize <- opt$minGS
maxGSSize <- opt$maxGS

model_dir <- file.path(base_dir, "model")
fig_dir   <- file.path(base_dir, "figures")

# --- 打印检查一下 (调试用) ---

if (!dir.exists(model_dir)) {
  stop(sprintf("Error! Can not find directory: %s", model_dir))
}

# 4. 创建图片保存目录
if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE) 
  print(paste("Created directory:", fig_dir))
}

# 5. 设置随机种子
set.seed(seed)

print("Setup Done. Starting analysis...")

fix_entrezID <- function(entrezIDs) {
  # 查找不符合标准格式的ID，即包含"c(...)"的元素
  for (i in 1:length(entrezIDs)) {
    if (grepl("^c\\(\"[0-9]+\", \"[0-9]+\"\\)$", entrezIDs[i])) {
      # 如果发现符合"c(\"7795\", \"51072\")"格式的元素, 进行拆分
      fixed_ids <- unlist(strsplit(gsub("^c\\(\"([0-9]+)\", \"([0-9]+)\"\\)$", "\\1,\\2", entrezIDs[i]), ","))
      # 替换原位置为拆分后的ID
      entrezIDs[i] <- fixed_ids[1]
      entrezIDs <- c(entrezIDs[1:i], fixed_ids[2:length(fixed_ids)], entrezIDs[(i + 1):length(entrezIDs)])
    }
  }
  return(entrezIDs)
}

FoldFunction<-function(results){
  library(stringr)
  gr1 <- as.numeric(str_split(results$GeneRatio,"/",simplify = T)[,1])
  gr2 <- as.numeric(str_split(results$GeneRatio,"/",simplify = T)[,2])
  bg1 <- as.numeric(str_split(results$BgRatio,"/",simplify = T)[,1])
  bg2 <- as.numeric(str_split(results$BgRatio,"/",simplify = T)[,2])
  results$fold <- (gr1/gr2)/(bg1/bg2)
  results$GeneRatio <- (gr1/gr2)
  return(results)
}

process_topic <- function(tp, sorted_row_indices, sizes, threshold = 100) {
  sorted_indices <- sorted_row_indices[, tp]
  sorted_sizes <- sizes[sorted_indices]
  cumulative_sizes <- cumsum(sorted_sizes)
  cutoff_index <- which(cumulative_sizes >= threshold)[1]
  if (is.na(cutoff_index)) {
    cutoff_index <- length(sorted_indices)
  }
  return(sorted_indices[1:cutoff_index])
}

mkGSEA<-function(df){
  df$Description<-lapply(df$Description,function(x){
    paste(unlist(strsplit(x,split = "_"))[-1],
          collapse = " ")%>% str_to_title()
  })%>%unlist()
  a<-gsub("In","in",df$Description)
  a<-gsub("Of","of",a)
  df$Description<-gsub("To","to",a)
  df<-df[order(df$NES,decreasing = T),]
  df$pval=as.numeric(df$pval)
  df$Description<-factor(df$Description,levels = unique(df$Description))
  rownames(df)=1:nrow(df)
  return(df)
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

process_gtp<-function(gtp){
  gtp_clean <- gtp[apply(gtp, 1, var) != 0, ]
  gtp_clean <- gtp_clean[complete.cases(gtp_clean), ]
  is_finite <- apply(gtp_clean, 1, function(x) all(is.finite(x)))
  gtp_clean <- gtp_clean[is_finite, ]
  print(paste("orignal gene number:", nrow(gtp), "gene number for enrichment:", nrow(gtp_clean)))
  return(gtp_clean)
}

enrichment_barplot<- function(results,tp){
  gop = ggplot(results[c(1:min(c(10,nrow(results)))),],aes(x=-log10(p.adjust),y=Description,fill=fold))+
    geom_bar(stat="identity",position="stack",width=0.6)+
    scale_fill_gradient2(low="#5BBCD6" ,mid = "white",high="#B40F20",midpoint = 0)+
    theme_bw()+theme(        axis.text.x = element_text(size = 12),
                             axis.text.y = element_text(size = 12))+
    theme(axis.line = element_line(colour = 'black', size = 0.5),
          plot.title = element_text(size = 20, hjust = 0.5),
          axis.title = element_text(size = 15, color = 'black'),
          panel.grid.major =element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          panel.border = element_blank(),
          title = element_text(size = 15),
          legend.text =element_text(size=10),  # Font size of legend labels.
          legend.title = element_text(size=12),
          legend.key.size=unit(0.2, "inches")
    )+labs(title ="",x = '-log10(adjust p value)',y = "")
  return(gop)
}

PLOT <-function(kk,tp,celltype,seed,geneList){
  res = kk@result%>%filter(qvalue < 0.05)
  write.csv(res,file=file.path(fig_dir,paste0(tp,"_",celltype,'_GOBP_seed',seed,".csv")),
            quote=F,row.names = F)
  if(nrow(res)>0){
    p1 <- cnetplot(kk,
                   showCategory = 5,
                   foldChange = geneList,
                   color_category="grey",
                   node_label="category",
                   circular = TRUE,
                   categorySize="pvalue")
    
    pdf(file=file.path(fig_dir,paste0(tp,"_",celltype,'_GOBP_seed',seed,"_cnetplot.pdf")),
        width =10,height =8)
    print(p1)
    dev.off()
    
    results<-FoldFunction(res)
    results<-results[order(results$p.adjust,decreasing = F),]
    results <-arrange(results,p.adjust,desc(fold))
    results$Description<-factor(results$Description,levels = rev(results$Description))
    
    pdf(file=file.path(fig_dir,paste0(tp,"_",celltype,'_GOBP_seed',seed,"_barplot.pdf")),width =8,height =5)
    print(enrichment_barplot(results,tp))
    dev.off()
  }
}


if(genome == "human"){
  library(org.Hs.eg.db)
  hs_msigdbr <- msigdbr(species="Homo sapiens")
  hsGO <- msigdbr(species="Homo sapiens",category="C5",subcategory = 'BP')
  hsKEGG <- msigdbr(species="Homo sapiens", category = "C2", subcategory = "KEGG")
}

if(genome == "mouse"){
  library(org.Mm.eg.db)
  hs_msigdbr <- msigdbr(species="Mus musculus")
  hsGO <- msigdbr(species="Mus musculus",category="C5",subcategory = 'BP')
  hsKEGG <- msigdbr(species="Mus musculus", category = "C2", subcategory = "KEGG")
}

if(genome=="zebrafish"){
  library(org.Dr.eg.db)
  hs_msigdbr <- msigdbr(species="Danio rerio")
  hsGO <- msigdbr(species="Danio rerio",category="C5",subcategory = 'BP')
  hsKEGG <- msigdbr(species="Danio rerio", category = "C2", subcategory = "KEGG")
}

gtp <-read.table(file.path(model_dir,"gene_topic_mat.txt"),sep = "\t",header = T,row.names = 1)
gtp_clean = process_gtp(gtp)
ctp <-read.table(file.path(model_dir,"topic_celltype_mat.txt"),sep = "\t",header = T,row.names = 1)
topic_number = nrow(ctp)
ctp = t(ctp)
colnames(ctp) = colnames(gtp)

pdf(file=file.path(fig_dir,'cell_component_heatmap.pdf'),width =6,height =5)
pc=pheatmap::pheatmap(ctp,
                      scale = "row",
                      show_colnames =T,
                      show_rownames = T,
                      cluster_rows = T,
                      border_color=NA,
                      cellwidth=15,
                      cellheight = 15
)
dev.off()

pdf(file=file.path(fig_dir,'gene_component_heatmap.pdf'),width =6,height =5)
pg=pheatmap::pheatmap(gtp_clean,
                      scale = "row",
                      show_colnames =T,
                      show_rownames = F,
                      cluster_rows = T,
                      border_color=NA,
)
dev.off()

pdf(file=file.path(fig_dir,'gene_component_Kmeans_heatmap.pdf'),width =6,height =5)
if(nrow(gtp_clean) >= topic_number) {
  pk <- pheatmap::pheatmap(gtp_clean,
                           scale = "row",
                           kmeans_k = topic_number,
                           show_colnames = T,
                           show_rownames = T,
                           cluster_rows = T,
                           border_color = NA,
                           cellwidth = 15,
                           cellheight = 15
  )
dev.off()

} else {
  message("错误: 清洗后的数据行数小于 topic_number，无法进行 K-means 聚类。")
}

# --- 1. 准备基础数据 ---
df_cluster <- data.frame(row.names = names(pk$kmeans$cluster), 
                         clustr = pk$kmeans$cluster)

# 获取所有 Topic 列表
all_topics <- colnames(pk$kmeans$centers)
sorted_row_indices <- apply(pk$kmeans$centers, 2, function(x) order(x, decreasing = TRUE))
topics_reached <- sapply(all_topics, process_topic, 
                         sorted_row_indices = sorted_row_indices, 
                         sizes = pk$kmeans$size)

# --- 2. 构建任务列表 (核心优化：合并两个循环的逻辑) ---
# 确定第一部分：有明确 CellType 的 Topic
max_col_indices <- apply(ctp, 1, which.max)
defined_celltypes <- names(max_col_indices)
defined_topics <- paste0("Topic", max_col_indices)

# 创建一个数据框来管理所有任务，包含: Topic名称, CellType名称
task_list <- data.frame(
  topic = defined_topics,
  celltype = defined_celltypes,
  stringsAsFactors = FALSE
)

# 确定第二部分：剩下的 (Unsure) Topic
remaining_topics <- setdiff(colnames(ctp), defined_topics)
if(length(remaining_topics) > 0){
  unsure_tasks <- data.frame(
    topic = remaining_topics,
    celltype = "unsure",
    stringsAsFactors = FALSE
  )
  task_list <- rbind(task_list, unsure_tasks)
}

# --- 3. 定义核心处理函数 ---
analyze_and_plot <- function(topic, celltype, df_cluster, topics_reached, gtp) {
  message(sprintf("Processing: %s (Type: %s)", topic, celltype))
  
  # 获取该 Topic 对应的 Clusters
  cls <- topics_reached[[topic]]
  if (is.null(cls) || length(cls) == 0) return(NULL)
  
  # 获取基因列表
  gene_symbols <- rownames(df_cluster)[df_cluster$clustr %in% cls]
  
  if (length(gene_symbols) == 0) {
    warning(paste("No genes found for topic:", topic))
    return(NULL)
  }
  # --- 构建 geneList 向量 (用于 cnetplot 上色) ---
  # 从 gtp 矩阵中提取该 Topic 的权重
  raw_scores <- gtp[, topic] 
  names(raw_scores) <- rownames(gtp)
  
  # 只保留当前 Cluster 包含的基因，并排序
  current_geneList <- raw_scores[gene_symbols]
  current_geneList <- sort(current_geneList, decreasing = TRUE)
  
  gene_map <- tryCatch({
    bitr(gene_symbols, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)
  }, error = function(e) { return(NULL) })
  
  if (is.null(gene_map) || nrow(gene_map) == 0) return(NULL)
  
  # 处理自定义的 fix_entrezID (保留你的逻辑)
  entrezIDs <- gene_map$ENTREZID %>% unique() %>% fix_entrezID()
  
  # GO 富集分析
  # 使用 tryCatch 防止某个 Topic 富集失败导致整个脚本崩溃
  kk <- tryCatch({
    enrichGO(gene = entrezIDs,
             OrgDb = org.Hs.eg.db,
             pvalueCutoff = 0.05,
             qvalueCutoff = 0.05,
             minGSSize = minGSSize,
             maxGSSize = maxGSSize,
             pAdjustMethod = "BH",
             ont = "BP",
             readable = TRUE)
  }, error = function(e) { message("Enrichment failed for ", topic); return(NULL) })
  
  if (!is.null(kk) && nrow(kk) > 0) {
    PLOT(kk,topic,celltype,seed,current_geneList) # 调用你的绘图函数
  }
}

# --- 4. 执行循环 ---
# 使用 lapply 或 for 循环遍历任务列表
for(i in 1:nrow(task_list)) {
  analyze_and_plot(
    topic = task_list$topic[i],
    celltype = task_list$celltype[i],
    df_cluster = df_cluster,
    topics_reached = topics_reached,
    gtp = gtp
  )
}
