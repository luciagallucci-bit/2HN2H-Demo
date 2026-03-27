# To Hit or Not To Hit: A Demo

# Authors:
# Marco Lattanzio
# Johannes Resin
# Donatella Firmani
# Lorenzo Balzotti 
# Gian Mario Sangiovanni,
# Lucia Gallucci
# Giovanna Jona Lasinio

# Brief Description: bla bla bla 

# install.packages(c("bslib","shinyWidgets"))
# install.packages("shinycssloaders")
# install.packages("arrow")

# Libraries

library(shiny)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(DT)
library(stringr)
library(bslib)
library(shinyWidgets)
library(arrow)
library(renv)
library(plotly)


# path <- paste("df_result_fold_", 0:14, "_YAGO3_10_RotatE.parquet", sep = "")
# path_hole <- paste("df_result_fold_", 0:14, "_YAGO3_10_HolE.parquet", sep = "")
# df_yago <- data.frame()
# str(dff)
# for (i in 1:length(path)) {
#   dff <- read_parquet(file = path[i])
#   dff$top_100_scores <- as.list(dff$top_100_scores)
#   dff$top_100_entities <- as.list(dff$top_100_entities)
#   dff$fold = i
#   df_yago <- rbind(df_yago, dff)
# }
# head(df_yago)
# 
# write_parquet(
#   df_yago,
#   sink = "combined_YAGO3_10_RotatE.parquet",
#   compression = "zstd",
#   compression_level = 10
# )

# 
# df_yago_hole <- data.frame()
# for (i in 1:length(path)) {
#   dff <- read_parquet(file = path_hole[i])
#   dff$top_100_scores <- as.list(dff$top_100_scores)
#   dff$top_100_entities <- as.list(dff$top_100_entities)
#   dff$fold = i
#   df_yago_hole <- rbind(df_yago_hole, dff)
# }
# head(df_yago_hole)
# 
# write_parquet(
#   df_yago_hole,
#   sink = "combined_YAGO3_10_HolE.parquet",
#   compression = "zstd",
#   compression_level = 10
# )
# 
# path_MurE <- paste("df_result_fold_", 0:14, "_YAGO3_10_MurE.parquet", sep = "")
# df_yago_MurE <- data.frame()
# for (i in 1:length(path)) {
#   dff <- read_parquet(file = path_MurE[i])
#   dff$top_100_scores <- as.list(dff$top_100_scores)
#   dff$top_100_entities <- as.list(dff$top_100_entities)
#   dff$fold = i
#   df_yago_MurE <- rbind(df_yago_MurE, dff)
# }
# head(df_yago_MurE)
# 
# write_parquet(
#   df_yago_MurE,
#   sink = "combined_YAGO3_10_MurE.parquet",
#   compression = "zstd",
#   compression_level = 10
# )

# 
# path_complex <- paste("df_result_fold_", 0:14, "_YAGO3_10_ComplEx.parquet", sep = "")
# df_yago_complex <- data.frame()
# for (i in 1:length(path)) {
#   dff <- read_parquet(file = path_complex[i])
#   dff$top_100_scores <- as.list(dff$top_100_scores)
#   dff$top_100_entities <- as.list(dff$top_100_entities)
#   dff$fold = i
#   df_yago_complex <- rbind(df_yago_complex, dff)
# }
# head(df_yago_complex)
# 
# write_parquet(
#   df_yago_complex,
#   sink = "combined_YAGO3_10_ComplEx.parquet",
#   compression = "zstd",
#   compression_level = 10
# )

# Main colors ------------------------------------------------------------------

hits_color <- "#2E86C1"
log_color  <- "#1F4E79"
accent     <- "#5DADE2"

# Utility functions ------------------------------------------------------------

softmax_from_logits <- function(logits) { # sum_exp trick to avoid numerical problems
  ex <- exp(logits - max(logits))
  ex / sum(ex)
}

# compute top-k log score using saved per-triple metadata:
# - softmax_true: numeric scalar = exp(score_true) / sum_exp_all
# - sum_exp_all: scalar sum(exp(all_logits))
# - top100_cumexp: numeric vector length 100 with cumulative sums of exp(scores_sorted)[1:k]
# - position: integer rank (1 = best)

compute_logk_from_meta <- function(position, softmax_true, sum_exp_all, top100_cumexp, n_entities, k) {
  
  if (k >= n_entities) { # this is just a check
    # if k covers all entities, the probabilistic top-k equals the model's distribution => -log(softmax_true)
    return(-log(max(softmax_true, 1e-300)))
  }
  
  if (position <= k) {
    # true tail inside top-k: probability equals its softmax
    return(-log(max(softmax_true, 1e-300))) 
  } 
  
  else {
    # true tail outside top-k: probability is uniform residual mass divided by (n_entities - k)
    # compute mass_topk from top100_cumexp if k <= 100
    
    if (k <= length(top100_cumexp)) {
      mass_topk <- top100_cumexp[k] / sum_exp_all
    } else {
      
      # k > length(top100_cumexp) (shouldn't happen since k <= 100 in UI), fall back to using top100 mass
      mass_topk <- top100_cumexp[length(top100_cumexp)] / sum_exp_all
      # remaining mass is (1 - mass_topk) spread over n_entities - k
    }
    
    pi_k <- (1 - mass_topk) / (n_entities - k)
    # numeric safety
    pi_k <- max(pi_k, 1e-300)
    return(-log(pi_k))
  }
} # this function is ok!

# Helper to compute mean and 95% CI across folds
summarize_across_folds <- function(df, value_col = fold) {
  df %>%
    group_by(value_col) %>% # here df is already per-fold per-triple aggregate; but we'll compute per-fold metrics first
    summarise(.groups = "drop") # value col is the name of the folder columns
}
#dati <- read.table("train_YAGO3-10.txt", blank.lines.skip = FALSE, col.names = c("head", "relation", "tail"))

# ---------- Data generator (datasets × 15 folds) ------------
#dati <- read.table(file ="train_WN18RR.txt", col.names = c("head", "relation", "tail"), 
#blank.lines.skip = FALSE)
#dati <- dati[1:100, ]
dataset_name = "YAGO3_10"
model_name = "MurE"
#dati = read_parquet(file = "combined_YAGO3_10_RotatE.parquet")
read_dataset_with_folds <- function(dataset_name,
                                    model_name) { # this is the function to read and preprocess correctly the dataset 
  
  # Returns a tibble with one row per (fold, triple) containing:
  # head, relation, tail, fold, position, score (logit of true tail),
  # softmax_true, top100_scores (numeric vector), top100_entities (char vector),
  # top100_cumexp (numeric cumexp vector length 100), sum_exp_all, n_entities
  ds <- as.character(dataset_name[1])
  md <- as.character(model_name[1])
  file_path <- paste0("combined_", ds, "_", md, ".parquet")
  #file_path <- paste("combined_", paste(dataset_name, sep = "", "_"), model_name, ".parquet", sep = "") # define the path of the parquet file
  train_path <- paste("train_", dataset_name, ".txt", sep = "") # training data --> i think it is not useful
  dati <- read_parquet(file_path) # import the corresponding data
  dati_train <- read.table(train_path, col.names = c("head", "relation", "tail"), 
                           blank.lines.skip = FALSE) 
  head <- as.character(dati$head) # head in test set
  relations <- as.character(dati$relation) # relations in test set
  tails <- as.character(dati$tail) # tails in test set
  n_entities <- length(c(tails, head)) # total number of entities inside the test set
  # it you want to use Chebyshev inequality, Recall fr(|X_{i} - \mu| > \sigma \epsilon) <= 1/\epsilon^2
  # set 1/epsilon^2 = 0.05 and obtaint
  eps <- sqrt(1/0.05)

  k_range <- c(1, 3, 10, 20)
  best_position <- matrix(NA, nrow = nrow(dati), ncol = 13) # here we work more at a tuple level 
  best_position <- as.data.frame(best_position)
  colnames(best_position) <- c("entity_position", "softmax_score", "fold", paste("k", k_range, sep = "_"), "mrr", "head", "relation", "tail", "top_100_scores", "top_100_entities")
  best_position[, "fold"] <- dati$fold
  best_position[, "head"] <- head
  best_position[, "tail"] <- tails
  best_position[, "relation"] <- relations
  best_position$top_100_scores <- dati$top_100_scores
  best_position$top_100_entities <- dati$top_100_entities
  for (i in 1:nrow(best_position)) {
    exp_score <- exp(dati$top_100_scores[i][[1]])/dati$sum_exp[i]
    if (tails[i] %in% dati$top_100_entities[i][[1]]) {
      best_position[i,1] <- dati$hits[i]
      best_position[i, 8] <- 1/dati$hits[i]
      #best_position[i,1] <- which(dati$top_100_entities[i][[1]] == tails[i])
      best_position[i, 2] <- exp(dati$top_100_scores[[i]][best_position[i, 1]])/(dati$sum_exp[i])
      best_position[i, 4:7] <- sapply(k_range, FUN = function(k){
        if (best_position[i,1] <= k) {
          -log(best_position[i, 2])
          
        }
        else{
          mass_topk <- cumsum(exp_score[1:k])[k]
          -log((1 - mass_topk) / (n_entities - k))
          
          
        }
        
      })
      
      
    }
    if (is.na(best_position[i, 1]) ) {
      best_position[i, 1] <- dati$hits[i]
      best_position[i, 8] <- 1/dati$hits[i]
      best_position[i, 4:7] <- sapply(k_range, FUN = function(k){
        mass_topk <- cumsum(exp_score[1:k])[k]
        -log((1 - mass_topk) / (n_entities - k))
        
        
      })
    }
    
  }
  dati |> 
    group_by(fold) |> 
    summarise(
      across(
        c(`hits@1`, `hits@3`, `hits@10`, `hits@20`, mrr, orig_brier_1, orig_brier_3, orig_brier_10, orig_brier_20),
        list(
          mean = ~mean(.x, na.rm = TRUE),
          # Chebyshev Lower Bound (capped at 0 minimum)
          ci_low = ~mean(.x, na.rm = TRUE) - 
            eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x)))),
          # Chebyshev Upper Bound (capped at 1 maximum)
          ci_high = ~mean(.x, na.rm = TRUE) + 
            eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x))))
        ),
        .names = "{.col}_{.fn}" 
      )
    ) -> results_matrix
  
  best_position |> 
    group_by(fold) |> 
    summarise(
      across(
        c("k_1", "k_3", "k_10", "k_20"),
        list(
          mean = ~mean(.x, na.rm = TRUE),
          # Chebyshev Lower Bound (capped at 0 minimum)
          ci_low = ~mean(.x, na.rm = TRUE) - 
            eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x)))),
          # Chebyshev Upper Bound (capped at 1 maximum)
          ci_high = ~mean(.x, na.rm = TRUE) + 
            eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x))))
        ),
        .names = "{.col}_{.fn}" 
      )
    ) -> results_matrix_bis

  results_matrix <- results_matrix %>%
    left_join(results_matrix_bis, by = "fold")
  dati |>
    select(orig_brier_1, orig_brier_3, orig_brier_10, orig_brier_20) -> dat_appoggio
  best_position <- cbind(best_position, dat_appoggio)

  output <- list(results_matrix = results_matrix, best_position = best_position)
  return(output)
}


# ---------- UI ------------
# ---------- UI ------------
ui <- fluidPage(
  theme = bs_theme(
    version = 5,
    bootswatch = "flatly",
    primary = "#1F4E79",
    secondary = "#2E86C1",
    base_font = font_google("Inter"),
    heading_font = font_google("Inter")
  ),
  
  tags$head(
    tags$style(HTML("
      body { background-color: #F7F9FC; }
      .title-panel { background: linear-gradient(90deg,#1F4E79,#2E86C1); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
      .card { border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; }
      .control-label { font-weight: 600; }
    "))
  ),
  
  div(class="title-panel",
      h2("To Hit or not to Hit: Knowledge Graph Completion Analysis"),
      p("Comparison of Proper Scores and Standard Metrics")
  ),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Configuration"),
      selectInput("dataset", "Dataset", choices = c("YAGO3_10", "WN18RR", "KINSHIPS", "FB15k237")),
      pickerInput(
        inputId = "model", 
        label = "Select Models to Compare", 
        choices = c("RotatE", "HolE", "MurE", "ComplEx"), 
        selected = "MurE", 
        options = list(`actions-box` = TRUE), 
        multiple = TRUE
      ),
      selectInput("k", "Top-k value", choices = c("1", "3", "10", "20"), selected = "10"),
      sliderInput("max_rank_filter", "Rank Filter (Cross-Fold):", min = 50, max = 1000, value = 500, step = 50),
      
      hr(),
      h5("Inspector Settings"),
      selectInput("model_single", "Model for Detailed Analysis", choices = c("RotatE", "HolE", "MurE", "ComplEx"), selected = "MurE"),
      selectInput("triple_filter", "Filter Triples By Behavior", 
                  choices = c("Medium (Rank near 50)", "Random (Mixed)", "Easy (Consistently High Rank)", "Hard (Consistently Low Rank)", "High Variance")),
      sliderInput("n_sample_triples", "Triples to inspect", min = 5, max = 80, value = 20),
      
      actionBttn("resample", "Fetch Triples", style = "gradient", color = "primary"),
      br(),
      downloadButton("download_combined_csv", "Download Current View")
    ),
    
    mainPanel(
      tabsetPanel(
        id = "tabs",
        
        # ---------- Tab 1: Model Evaluation ----------
        tabPanel("Model Evaluation",
                 br(),
                 fluidRow(
                   column(6, card(
                     card_header("Standard & Brier Score (0-1 Scale)"), 
                     card_body(shinycssloaders::withSpinner(plotOutput("unified_metrics_plot", height = 450)))
                   )),
                   column(6, card(
                     card_header(textOutput("logk_title")), 
                     card_body(shinycssloaders::withSpinner(plotOutput("logk_forest", height = 450)))
                   ))
                 )
        ),
        
        # ---------- Tab 2: Cross-Fold Analysis (Lorenzo Plots) ----------
        tabPanel("Cross-Fold Analysis",
                 br(),
                 fluidRow(
                   column(3, card(
                     card_header("Settings"),
                     card_body(
                       selectInput("ref_fold", "Reference Fold", choices = as.character(1:15), selected = "1"),
                       selectInput("crossfold_metric", "Metric for Sorting", choices = c("Log-K" = "k", "Brier Score" = "brier"), selected = "k"),
                       checkboxInput("random_fold", "Use random fold", value = FALSE)
                     )
                   )),
                   column(9, 
                          card(
                            card_header("Rank Distribution Stability"),
                            card_body(shinycssloaders::withSpinner(plotOutput("crossfold_logk", height = 400)))
                          ),
                          card(
                            card_header("Rank Distribution vs MRR"),
                            card_body(shinycssloaders::withSpinner(plotOutput("crossfold_mrr", height = 400)))
                          )
                   )
                 )
        ),
        
        # ---------- Tab 3: Triple Inspector ----------
        tabPanel("Triple Inspector",
                 br(),
                 card(
                   card_header("Sampled Triples Summary"), 
                   card_body(DTOutput("triples_table_summary"))
                 ),
                 fluidRow(
                   column(5, card(
                     card_header("Prediction Details"), 
                     card_body(uiOutput("selected_triple_ui"), DTOutput("triple_fold_table"))
                   )),
                   column(7, card(
                     card_header("Tuple-Level Variance Across Folds"), 
                     card_body(
                       fluidRow(
                         column(6, plotOutput("tuple_ci_plot_logk", height = 250)),
                         column(6, plotOutput("tuple_ci_plot_mrr", height = 250))
                       )
                     )
                   ))
                 )
        )
      )
    )
  )
)
# ---------- Server ------------
server <- function(input, output, session) {
  
  # Title for Log-K plot
  output$logk_title <- renderText({ paste0("Mean Top-", input$k, " Log Score Comparison") })
  
  # Reactive Data Loading
  processed_data <- reactive({
    req(input$dataset, input$model)
    all_models_list <- map(input$model, function(m) {
      tryCatch({
        res <- read_dataset_with_folds(input$dataset, m)
        res$results_matrix$model <- m
        res$best_position$model <- m
        return(res)
      }, error = function(e) { return(NULL) })
    })
    all_models_list <- compact(all_models_list)
    if(length(all_models_list) == 0) return(NULL)
    
    list(
      results_matrix = bind_rows(map(all_models_list, "results_matrix")),
      best_position = bind_rows(map(all_models_list, "best_position"))
    )
  })
  
  res_matrix <- reactive({ req(processed_data()$results_matrix) })
  best_pos   <- reactive({ req(processed_data()); processed_data()$best_position })
  
  
  res_matrix_single <- reactive({
    req(res_matrix(), input$model_single)
    res_matrix() %>% filter(model == input$model_single)
  })
  best_pos_single <- reactive({
    req(best_pos(), input$model_single)
    best_pos() %>% filter(model == input$model_single)
  })
  
  bp_sampled_reactive <- reactive({
    req(best_pos_single(), sampled_triples())
    best_pos_single() %>%
      semi_join(sampled_triples(), by = c("head", "relation", "tail"))
  })
  
  
  # --- Hits & MRR Plot ---
  output$unified_metrics_plot <- renderPlot({
    req(res_matrix())
    df <- res_matrix()
    k_val <- input$k
    
    # 1. Definizione stringhe nomi colonne (devono essere stringhe per il confronto)
    hits_mean_str  <- paste0("hits@", k_val, "_mean")
    hits_low_col   <- paste0("hits@", k_val, "_ci_low")
    hits_high_col  <- paste0("hits@", k_val, "_ci_high")
    
    brier_mean_str <- paste0("orig_brier_", k_val, "_mean")
    brier_low_col  <- paste0("orig_brier_", k_val, "_ci_low")
    brier_high_col <- paste0("orig_brier_", k_val, "_ci_high")
    
    # 2. Pivot e Mutate
    plot_data <- df %>%
      pivot_longer(
        cols = c(all_of(hits_mean_str), mrr_mean, all_of(brier_mean_str)),
        names_to = "metric_type",
        values_to = "mean_val"
      ) %>%
      mutate(
        # Corretto: mettiamo i nomi delle metriche tra virgolette per il confronto
        ci_low = case_when(
          metric_type == "mrr_mean"       ~ mrr_ci_low,
          metric_type == hits_mean_str    ~ .data[[hits_low_col]],
          metric_type == brier_mean_str   ~ .data[[brier_low_col]],
          TRUE ~ NA_real_
        ),
        ci_high = case_when(
          metric_type == "mrr_mean"       ~ mrr_ci_high,
          metric_type == hits_mean_str    ~ .data[[hits_high_col]],
          metric_type == brier_mean_str   ~ .data[[brier_high_col]],
          TRUE ~ NA_real_
        ),
        # Label pulite per la legenda
        metric_label = case_when(
          metric_type == "mrr_mean"       ~ "MRR",
          metric_type == hits_mean_str    ~ paste0("Hits@", k_val),
          metric_type == brier_mean_str   ~ "Brier Score",
          TRUE ~ metric_type
        )
      )
    
    # 3. Plot
    ggplot(plot_data, aes(x = factor(fold), y = mean_val, color = model, shape = metric_label)) +
      geom_point(position = position_dodge(width = 0.8), size = 3.5) +
      geom_errorbar(aes(ymin = ci_low, ymax = ci_high), 
                    position = position_dodge(width = 0.8), width = 0.4, alpha = 0.6) +
      scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.1)) +
      labs(
        title = "Accuracy & Calibration Comparison",
        subtitle = paste0("Metrics evaluated at k = ", k_val),
        x = "Fold", 
        y = "Value (0-1 Scale)", 
        color = "Model", 
        shape = "Metric"
      ) +
      theme_minimal() +
      theme(legend.position = "bottom")
  })
  
  # --- Log-K Plot ---
  output$logk_forest <- renderPlot({
    req(res_matrix())
    k_val <- input$k
    
    # Column naming logic from your read_dataset function
    mean_col <- paste0("k_", k_val, "_mean")
    low_col  <- paste0("k_", k_val, "_ci_low")
    high_col <- paste0("k_", k_val, "_ci_high")
    
    ggplot(res_matrix(), aes(x = factor(fold), y = .data[[mean_col]], color = model)) +
      geom_point(position = position_dodge(width = 0.5), size = 3) +
      geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[high_col]]), 
                    position = position_dodge(width = 0.5), width = 0.4) +
      labs(x = "Fold", y = "Log Score (Proper Score)", color = "Model") +
      theme_minimal() +
      theme(legend.position = "bottom")
  })
  # --- Resampling Filtered Triples ---
  sampled_triples <- eventReactive(input$resample, {
    req(best_pos_single())
    set.seed(input$seed)
    
    # Calculate stats per triple across folds
    triple_stats <-best_pos_single() %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_rank = mean(entity_position, na.rm = TRUE),
        sd_rank = sd(entity_position, na.rm = TRUE),
        .groups = "drop"
      )
    
    # Apply user-selected filter
    if (input$triple_filter == "Easy (Consistently High Rank)") {
      valid_triples <- triple_stats %>% filter(mean_rank <= 5)
    } else if (input$triple_filter == "Hard (Consistently Low Rank)") {
      valid_triples <- triple_stats %>% filter(mean_rank >= 70)
    } else if (input$triple_filter == "High Variance (Model Uncertain)") {
      valid_triples <- triple_stats %>% arrange(desc(sd_rank)) %>% head(150)
    } else if (input$triple_filter == "Random (Mixed)") {
      valid_triples <- triple_stats 
    }  else if (input$triple_filter == "Medium (Rank near 50)") {
      # Safest approach: grab top 150 closest to rank 50, then sample from them
      valid_triples <- triple_stats %>% arrange(abs(mean_rank - 50)) %>% head(150)
    }
    # Sample the requested number (or the max available if fewer match the filter)
    n <- min(nrow(valid_triples), input$n_sample_triples)
    valid_triples %>% slice_sample(n = n)
  }, ignoreNULL = FALSE)
  
  # --- Triples summary table (Rank, MRR, LogK) ---
  output$triples_table_summary <- renderDT({
    req(best_pos_single(), sampled_triples(), input$k)
    
    eps <- sqrt(1/0.05) 
    k_col <- paste0("k_", input$k)
    
    stat <- best_pos_single() %>%
      mutate(mrr = 1 / entity_position) %>%
      group_by(head, relation, tail) %>%
      summarise(
        n_folds = sum(!is.na(entity_position)),
        mean_pos = round(mean(entity_position, na.rm=TRUE), 2),
        mean_mrr = round(mean(mrr, na.rm=TRUE), 4),
        mean_logk = mean(.data[[k_col]], na.rm=TRUE),
        sd_logk = sd(.data[[k_col]], na.rm=TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        sd_logk = ifelse(is.na(sd_logk), 0, sd_logk),
        logk_low = pmax(0, mean_logk - eps * (sd_logk / sqrt(n_folds))),
        logk_high = mean_logk + eps * (sd_logk / sqrt(n_folds))
      ) %>%
      inner_join(sampled_triples(), by = c("head", "relation", "tail")) %>%
      arrange(mean_pos) %>%
      mutate(
        LogK_CI = sprintf("%.3f [%.3f, %.3f]", mean_logk, logk_low, logk_high)
      ) %>%
      select(head, relation, tail, mean_pos, mean_mrr, LogK_CI) %>%
      rename(`Mean Rank` = mean_pos, `Mean MRR` = mean_mrr, !!paste0("Log-", input$k, " CI") := LogK_CI)
    
    datatable(stat, options = list(pageLength = 5, scrollX = TRUE))
  })
  
  # --- UI for selecting a triple ---
  output$selected_triple_ui <- renderUI({
    req(sampled_triples())
    sampled <- sampled_triples()
    choices <- paste0(seq_len(nrow(sampled)), ": ", sampled$head, " | ", sampled$relation, " | ", sampled$tail)
    selectInput("selected_triple_idx", "Choose triple (from sample)", choices = choices, selected = choices[1])
  })
  
  # --- Fold Details Table (Showing Rank, MRR, Hits@k + All Log-K scores) ---
  output$triple_fold_table <- renderDT({
    req(input$selected_triple_idx, best_pos_single(), input$k)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    k_val <- as.numeric(input$k)
    
    triple_rows <- best_pos_single() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(
        MRR = round(1 / entity_position, 4),
        Hit_k = ifelse(entity_position <= k_val, 1, 0)
      ) %>%
      arrange(fold) %>%
      select(fold, entity_position, MRR, Hit_k, k_1, k_3, k_10, k_20) %>%
      mutate(across(starts_with("k_"), ~round(.x, 4)))
    
    datatable(triple_rows, options = list(pageLength = 5, scrollX = TRUE), 
              colnames = c("Fold", "Rank", "MRR", paste0("Hits@", k_val), "Log-1", "Log-3", "Log-10", "Log-20"))
  })
  
  # --- Tuple CI Plot 1: LOG-K SCORES ---
  output$tuple_ci_plot_logk <- renderPlot({
    req(input$selected_triple_idx, best_pos_single())
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    eps <- sqrt(1/0.05)
    
    tuple_data <-best_pos_single()%>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      select(fold, k_1, k_3, k_10, k_20)
    
    n_folds <- nrow(tuple_data)
    
    plot_df <- tuple_data %>%
      pivot_longer(cols = starts_with("k_"), names_to = "k_val", values_to = "score") %>%
      group_by(k_val) %>%
      summarise(Mean = mean(score, na.rm=TRUE), SD = sd(score, na.rm=TRUE), .groups = "drop") %>%
      mutate(
        SD = ifelse(is.na(SD), 0, SD),
        CI_Low = pmax(0, Mean - eps * (SD / sqrt(n_folds))),
        CI_High = Mean + eps * (SD / sqrt(n_folds)),
        k_num = as.numeric(gsub("k_", "", k_val)),
        Metric = factor(paste0("k=", k_num), levels = paste0("k=", c(1, 3, 10, 20, 50, 100)))
      )
    
    ggplot(plot_df, aes(x = Metric, y = Mean, color = Metric)) +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = CI_Low, ymax = CI_High), width = 0.2, linewidth = 1) +
      scale_color_viridis_d(option = "plasma", end = 0.8) +
      labs(title = "Top-K Log Scores", x = "k Value", y = "Log Score") +
      theme(legend.position = "none")
  })
  
  # --- Tuple CI Plot 2: MRR & Hits@k (UPDATED) ---
  output$tuple_ci_plot_mrr <- renderPlot({
    req(input$selected_triple_idx, best_pos_single())
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    eps <- sqrt(1/0.05)
    
    tuple_data <-best_pos_single() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(
        MRR = 1 / entity_position,
        `Hits@1` = as.numeric(entity_position <= 1),
        `Hits@3` = as.numeric(entity_position <= 3),
        `Hits@10` = as.numeric(entity_position <= 10),
        `Hits@20` = as.numeric(entity_position <= 20)
      )
    
    n_folds <- nrow(tuple_data)
    
    plot_df <- data.frame(
      Metric = factor(c("Hits@1", "Hits@3", "Hits@10", "Hits@20", "MRR"), 
                      levels = c("Hits@1", "Hits@3", "Hits@10", "Hits@20", "MRR")),
      Mean = c(mean(tuple_data$`Hits@1`), mean(tuple_data$`Hits@3`), mean(tuple_data$`Hits@10`), 
               mean(tuple_data$`Hits@20`), 
               mean(tuple_data$MRR)),
      SD = c(sd(tuple_data$`Hits@1`), sd(tuple_data$`Hits@3`), sd(tuple_data$`Hits@10`), 
             sd(tuple_data$`Hits@20`), 
             sd(tuple_data$MRR))
    ) %>%
      mutate(
        SD = ifelse(is.na(SD), 0, SD),
        CI_Low = pmax(0, Mean - eps * (SD / sqrt(n_folds))),
        CI_High = pmin(1, Mean + eps * (SD / sqrt(n_folds)))
      )
    
    ggplot(plot_df, aes(x = Metric, y = Mean, color = Metric)) +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = CI_Low, ymax = CI_High), width = 0.2, linewidth = 1) +
      # Expanded color palette to comfortably hold 7 metrics
      scale_color_manual(values = c("#9B59B6", "#3498DB", "#1ABC9C", "#F1C40F", "#E67E22", "#E74C3C", "#27AE60")) +
      ylim(0, 1) +
      labs(title = "MRR & Hits@k", x = "Metric", y = "Score [0,1]") +
      theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1) # Prevents text overlap now that we have more columns
      )
  })
  
  # --- Top 100 predictions ---
  output$top100_table <- renderDT({
    req(input$selected_triple_idx, best_pos_single(), input$fold_for_top100)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    fold_sel <- as.integer(input$fold_for_top100)
    
    row <- best_pos_single() %>% filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail, fold == fold_sel)
    if (nrow(row) == 0) return(datatable(tibble(note = "No data"), options = list(dom = 't')))
    
    ents <- unlist(row$top_100_entities[[1]])
    scs  <- unlist(row$top_100_scores[[1]])
    tbl <- tibble(pos = seq_along(ents), entity = ents, score = round(scs, 4))
    datatable(tbl, options = list(pageLength = 5, scrollX = TRUE))
  })
  
  output$download_combined_csv <- downloadHandler(
    filename = function() paste0("metrics_", input$dataset, "_", input$model, "_", Sys.Date(), ".csv"),
    content = function(file) { write.csv(res_matrix_single(), file, row.names = FALSE) }
  )
  
  # -- Lorenzo plot: fold selection --
  ref_fold_selected <- reactive({
    if (isTRUE(input$random_fold)) {
      sample(unique(best_pos_single()$fold), 1)
    } else {
      as.integer(input$ref_fold)
    }
  })
  
  crossfold_metric_col <- reactive({
    req(input$crossfold_metric, input$k)
    paste0(input$crossfold_metric, "_", input$k)
  })
  
  output$crossfold_logk <- renderPlot({
    req(bp_sampled_reactive(), input$max_rank_filter, input$k)
    
    bp_s <- bp_sampled_reactive() 
    ref_f <- ref_fold_selected()
    
    # Mapping metrica
    k_col <- case_when(
      input$crossfold_metric == "k"     ~ paste0("k_", input$k),
      input$crossfold_metric == "brier" ~ paste0("orig_brier_", input$k),
      TRUE ~ paste0("k_", input$k)
    )
    
    # 1. Pulizia dati: Se il Brier è > 1, lo normalizziamo (es. dividendo per N entità)
    # O semplicemente lo segnaliamo. Qui forziamo il valore nel range 0-1 se è Brier.
    ref_data <- bp_s %>% 
      filter(fold == ref_f) %>%
      select(head, relation, tail, ref_score = .data[[k_col]]) %>%
      mutate(ref_score = if(input$crossfold_metric == "brier" && any(ref_score > 1)) 
        ref_score / max(ref_score, na.rm=TRUE) else ref_score) %>%
      arrange(ref_score)
    
    # 2. Creazione del plot con asse X ordinato
    merged <- bp_s %>% 
      inner_join(ref_data, by = c("head", "relation", "tail")) %>%
      mutate(ref_score_label = factor(round(ref_score, 4), 
                                      levels = unique(round(ref_data$ref_score, 4))))
    
    ggplot(merged, aes(x = ref_score_label, y = entity_position)) +
      geom_boxplot(
        fill = ifelse(input$crossfold_metric == "brier", "#E67E22", "#2E86C1"), 
        alpha = 0.5, outlier.size = 1
      ) +
      scale_y_reverse(limits = c(input$max_rank_filter, 1)) +
      labs(
        title = paste0("Rank vs ", input$crossfold_metric),
        x = "Value ",
        y = "Rank"
      ) +
      theme_minimal() + 
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 8))
  })
  
  
  # -- Lorenzo plot: scatter cross-fold --
  outputOptions(output, "crossfold_logk", suspendWhenHidden = TRUE)

  # -- MRR plot --
  output$crossfold_mrr <- renderPlot({
    req(bp_sampled_reactive(), input$max_rank_filter)
    ref_fold <- ref_fold_selected()
    bp_s     <- bp_sampled_reactive()
    
    # 1. Prepare reference data 
    # Sort MRR from highest (1.0) to lowest (e.g., 0.001) using desc()
    ref_data <- bp_s %>%
      filter(fold == ref_fold) %>%
      mutate(ref_mrr = 1 / entity_position) %>%
      select(head, relation, tail, ref_mrr) %>%
      arrange(desc(ref_mrr)) %>%  # <-- AGGIUNTO desc() QUI
      mutate(idx = row_number()) # Assigns X-axis order based on the sort
    
    # Mappiamo l'indice al valore MRR arrotondato
    mrr_labels <- setNames(sprintf("%.3f", ref_data$ref_mrr), ref_data$idx)
    
    # 2. Merge back to the sampled data to get distributions across all folds
    merged <- bp_s %>%
      inner_join(ref_data, by = c("head", "relation", "tail"))
    
    # 3. Plot with standard Reverse Scale
    ggplot(merged, aes(x = factor(idx), y = entity_position)) +
      geom_boxplot(width = 0.5, outlier.size = 1, fill = "#27AE60", alpha = 0.6) +
      scale_y_reverse(limits = c(input$max_rank_filter, 1)) +
      # Applichiamo le etichette reali dell'MRR
      scale_x_discrete(labels = mrr_labels) +
      labs(
        title    = "Rank Distribution vs MRR",
        subtitle = paste0("Reference Fold = ", ref_fold, " | Ordered Best to Worst"),
        x        = "Reference MRR (1.0 \u2192 0.0)",
        y        = "Rank (Inverted)"
      ) +
      theme_minimal() +
      theme(
        axis.text.x      = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 9),
        axis.ticks.x     = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position  = "none"
      )
  })
}

shinyApp(ui = ui, server = server)