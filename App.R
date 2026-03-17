# To Hit or Not To Hit: A Demo
# Authors:
# Marco Lattanzi
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

# path <- paste("YAGO RotatE/df_result_fold_", 0:14, "_YAGO3_10_RotatE.parquet", sep = "")
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
model_name = "RotatE"
#dati = read_parquet(file = "combined_YAGO3_10_RotatE.parquet")
read_dataset_with_folds <- function(dataset_name,
                                    model_name) { # this is the function to read and preprocess correctly the dataset 
  
  # Returns a tibble with one row per (fold, triple) containing:
  # head, relation, tail, fold, position, score (logit of true tail),
  # softmax_true, top100_scores (numeric vector), top100_entities (char vector),
  # top100_cumexp (numeric cumexp vector length 100), sum_exp_all, n_entities
  file_path <- paste("combined_", paste(dataset_name, sep = "", "_"), model_name, ".parquet", sep = "") # define the path of the parquet file
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
  dati |> 
    group_by(fold) |> 
    summarise(
      across(
        c(`hits@1`, `hits@3`, `hits@10`, `hits@20`, `hits@50`, `hits@100`, mrr, log_1_sparse, log_1_softmax,  
          log_3_sparse, log_3_softmax, log_10_sparse, log_10_softmax, log_20_sparse, log_20_softmax, log_50_sparse, 
          log_50_softmax, log_100_sparse, log_100_softmax),
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
  k_range <- c(1, 3, 10, 20, 50, 100)
  best_position <- matrix(NA, nrow = nrow(dati), ncol = 14) # here we work more at a tuple level 
  best_position <- as.data.frame(best_position)
  colnames(best_position) <- c("entity_position", "softmax_score", "fold", paste("k", k_range, sep = "_"), "head", "relation", "tail", "top_100_scores", "top_100_entities")
  best_position[, "fold"] <- dati$fold
  best_position[, "head"] <- head
  best_position[, "tail"] <- tails
  best_position[, "relation"] <- relations
  best_position$top_100_scores <- dati$top_100_scores
  best_position$top_100_entities <- dati$top_100_entities
  for (i in 1:nrow(best_position)) {
    if (tails[i] %in% dati$top_100_entities[i][[1]]) {
      best_position[i,1] <- which(dati$top_100_entities[i][[1]] == tails[i])
      best_position[i, 2] <- exp(dati$top_100_scores[[i]][best_position[i, 1]])/(dati$sum_exp[i])
      exp_score <- exp(dati$top_100_scores[i][[1]])/dati$sum_exp[i]
      best_position[i, 4:9] <- sapply(k_range, FUN = function(k){
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
      best_position[i, 1] <- 101
      best_position[i, 4:9] <- sapply(k_range, FUN = function(k){
        mass_topk <- cumsum(exp_score[1:k])[k]
        -log((1 - mass_topk) / (n_entities - k))
        
        
      })
    }
    
    
  }
  
  output <- list(results_matrix = results_matrix, best_position = best_position)
  return(output)
}


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
    .card { border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); }
    .control-label { font-weight: 600; }
    "))
  ),
  
  div(class="title-panel",
      h2("To Hit or not to Hit"),
      p("Interactive demo")
  ),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 3,
      h4("Configuration"),
      
      selectInput("dataset", "Dataset", choices = c("YAGO3_10")),
      selectInput("model", "Model", choices = c("RotatE")),
      selectInput("k", "Top-k value (for Hits & LogK)", choices = c("1", "3", "10", "20", "50", "100"), selected = "10"),
      
      hr(),
      h5("Inspector Settings"),
      selectInput("triple_filter", "Filter Triples By Behavior", 
                  choices = c("Medium (Rank near 50)", "Random (Mixed)", "Easy (Consistently High Rank)", "Hard (Consistently Low Rank)", "High Variance (Model Uncertain)")),
      sliderInput("n_sample_triples", "Triples to inspect", min = 5, max = 80, value = 20),
      numericInput("seed", "Random seed", value = 42),
      
      actionBttn("resample", "Fetch Triples", style = "gradient", color = "primary"),
      helpText(""),
      br(),
      
      radioGroupButtons(
        inputId = "view_mode",
        label = "Evaluation View Mode",
        choices = c("Per Fold", "Overall Average"),
        selected = "Per Fold",
        status = "primary",
        justified = TRUE
      ),
      br(),
      downloadButton("download_combined_csv", "Download Current View")
    ),
    
    mainPanel(
      tabsetPanel(
        id="tabs",
        tabPanel("Model Evaluation",
                 br(),
                 fluidRow(
                   column(12, card(card_header("Hits@k & MRR Performance"), 
                                   card_body(shinycssloaders::withSpinner(plotOutput("unified_metrics_plot", height=400)))))
                 ),
                 br(),
                 fluidRow(
                   column(6, card(card_header("Top-k Log Score"), 
                                  card_body(shinycssloaders::withSpinner(plotOutput("logk_forest", height=350))))),
                   column(6, card(card_header("Rank vs Top-k Log Score"), 
                                  card_body(shinycssloaders::withSpinner(plotOutput("position_vs_logk", height=350)))))
                 )
        ),
        tabPanel("Triple Inspector",
                 br(),
                 fluidRow(
                   column(12, card(card_header("Sampled Triples Summary (with Tuple-Level Chebyshev CIs)"), 
                                   card_body(DTOutput("triples_table_summary"))))
                 ),
                 fluidRow(
                   column(5, card(card_header("Prediction Details Across Folds"), card_body(
                     uiOutput("selected_triple_ui"), br(),
                     DTOutput("triple_fold_table")
                   ))),
                   column(7, card(card_header("Tuple-Level Variance Across Folds"), card_body(
                     fluidRow(
                       column(6, plotOutput("tuple_ci_plot_logk", height=250)),
                       column(6, plotOutput("tuple_ci_plot_mrr", height=250))
                     ),
                     br(),
                     selectInput("fold_for_top100", "Fold for Top-100 Predictions", choices = as.character(1:15)),
                     DTOutput("top100_table")
                   )))
                 )
        )
      )
    )
  )
)

# ---------- Server ------------

server <- function(input, output, session) {
  
  theme_set(
    theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(face="bold", size=14),
        axis.title = element_text(size=12),
        panel.grid.minor = element_blank()
      )
  )
  
  # Reactive Data Loading
  processed_data <- reactive({
    req(input$dataset, input$model)
    tryCatch({
      read_dataset_with_folds(input$dataset, input$model)
    }, error = function(e) {
      showNotification(paste("Data not found for", input$dataset, input$model), type = "error")
      return(NULL)
    })
  })
  
  res_matrix <- reactive({ req(processed_data()); processed_data()$results_matrix })
  best_pos <- reactive({ req(processed_data()); processed_data()$best_position })
  
  # --- Unified Forest Plot: Hits@k and MRR ---
  output$unified_metrics_plot <- renderPlot({
    req(res_matrix())
    mat <- res_matrix()
    k_val <- input$k
    hits_mean_col <- paste0("hits@", k_val, "_mean")
    hits_low_col <- paste0("hits@", k_val, "_ci_low")
    hits_high_col <- paste0("hits@", k_val, "_ci_high")
    
    if (input$view_mode == "Per Fold") {
      plot_data <- data.frame(
        fold = rep(mat$fold, 2),
        metric = c(rep(paste0("Hits@", k_val), nrow(mat)), rep("MRR", nrow(mat))),
        mean = c(mat[[hits_mean_col]], mat$mrr_mean),
        ci_low = c(mat[[hits_low_col]], mat$mrr_ci_low),
        ci_high = c(mat[[hits_high_col]], mat$mrr_ci_high)
      )
      
      ggplot(plot_data, aes(x = factor(fold), y = mean, color = metric)) +
        geom_point(position = position_dodge(width = 0.5), size = 3) +
        geom_errorbar(aes(ymin = ci_low, ymax = ci_high), 
                      position = position_dodge(width = 0.5), width = 0.3, linewidth = 0.8) +
        scale_color_manual(values = c("#1F4E79", "#27AE60")) +
        labs(title = paste0("Hits@", k_val, " and MRR per Fold"), x = "Fold", y = "Score") +
        theme(legend.position = "bottom", legend.title = element_blank())
      
    } else {
      # Overall Chebyshev Bounds across folds
      n_folds <- nrow(mat)
      eps <- sqrt(1/0.05)
      
      hits_mean <- mean(mat[[hits_mean_col]], na.rm=TRUE)
      hits_sd <- sd(mat[[hits_mean_col]], na.rm=TRUE)
      hits_ci_low <- max(0, hits_mean - eps * (hits_sd / sqrt(n_folds)))
      hits_ci_high <- min(1, hits_mean + eps * (hits_sd / sqrt(n_folds)))
      
      mrr_mean <- mean(mat$mrr_mean, na.rm=TRUE)
      mrr_sd <- sd(mat$mrr_mean, na.rm=TRUE)
      mrr_ci_low <- max(0, mrr_mean - eps * (mrr_sd / sqrt(n_folds)))
      mrr_ci_high <- min(1, mrr_mean + eps * (mrr_sd / sqrt(n_folds)))
      
      plot_data <- data.frame(
        metric = factor(c(paste0("Hits@", k_val), "MRR"), levels = c(paste0("Hits@", k_val), "MRR")),
        mean = c(hits_mean, mrr_mean),
        ci_low = c(hits_ci_low, mrr_ci_low),
        ci_high = c(hits_ci_high, mrr_ci_high)
      )
      
      ggplot(plot_data, aes(x = metric, y = mean, color = metric)) +
        geom_point(size = 5) +
        geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.15, linewidth = 1) +
        ylim(0, 1) +
        scale_color_manual(values = c("#1F4E79", "#27AE60")) +
        labs(title = paste0("Overall Mean: Hits@", k_val, " & MRR (Across 15 Folds)"), 
             subtitle = "Error bars indicate 95% Chebyshev CI",
             x = "Metric", y = "Score") +
        theme(legend.position = "none")
    }
  })
  
  # --- Forest Plot: LogK ---
  output$logk_forest <- renderPlot({
    req(res_matrix())
    mat <- res_matrix()
    k_val <- input$k
    mean_col <- paste0("log_", k_val, "_softmax_mean")
    low_col <- paste0("log_", k_val, "_softmax_ci_low")
    high_col <- paste0("log_", k_val, "_softmax_ci_high")
    
    if (input$view_mode == "Per Fold") {
      ggplot(mat, aes(x = factor(fold), y = .data[[mean_col]])) +
        geom_point(color = "#E67E22", size = 3) +
        geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[high_col]]), width = 0.2, color = "#E67E22", linewidth = 0.8) +
        geom_hline(aes(yintercept = mean(.data[[mean_col]], na.rm=TRUE)), color = "red", linetype = "dashed") +
        labs(title = paste0("Mean top-", k_val, " Log Score"), x = "Fold", y = "Log Score")
    } else {
      # Overall Chebyshev Bounds for Log-K
      n_folds <- nrow(mat)
      eps <- sqrt(1/0.05)
      
      overall_mean <- mean(mat[[mean_col]], na.rm=TRUE)
      overall_sd <- sd(mat[[mean_col]], na.rm=TRUE)
      ci_low <- max(0, overall_mean - eps * (overall_sd / sqrt(n_folds)))
      ci_high <- overall_mean + eps * (overall_sd / sqrt(n_folds))
      
      ggplot(data.frame(x="Overall", y=overall_mean, low=ci_low, high=ci_high), aes(x=x, y=y)) +
        geom_point(color = "#E67E22", size = 5) +
        geom_errorbar(aes(ymin = low, ymax = high), width = 0.1, color = "#E67E22", linewidth = 1) +
        labs(title = paste0("Overall Mean top-", k_val, " Log Score"), 
             subtitle = "Error bars indicate 95% Chebyshev CI",
             x = "Dataset Avg", y = "Log Score")
    }
  })
  
  # --- Position vs LogK Score ---
  output$position_vs_logk <- renderPlot({
    req(best_pos(), input$k)
    bp <- best_pos()
    k_col <- paste0("k_", input$k)
    
    stat <- bp %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_position = mean(entity_position, na.rm=TRUE), 
        mean_logk = mean(.data[[k_col]], na.rm=TRUE), 
        .groups = "drop"
      )
    
    ggplot(stat, aes(x = mean_position, y = mean_logk)) +
      geom_jitter(alpha = 0.4, height = 0, width = 0.6, color="#8E44AD") +
      labs(title = paste0("Mean rank vs Mean Top-", input$k, " Log Score"), x = "Mean Rank", y = paste0("Mean Top-", input$k, " Log Score"))
  })
  
  # --- Resampling Filtered Triples ---
  sampled_triples <- eventReactive(input$resample, {
    req(best_pos())
    set.seed(input$seed)
    
    # Calculate stats per triple across folds
    triple_stats <- best_pos() %>%
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
    req(best_pos(), sampled_triples(), input$k)
    
    eps <- sqrt(1/0.05) 
    k_col <- paste0("k_", input$k)
    
    stat <- best_pos() %>%
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
    req(input$selected_triple_idx, best_pos(), input$k)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    k_val <- as.numeric(input$k)
    
    triple_rows <- best_pos() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(
        MRR = round(1 / entity_position, 4),
        Hit_k = ifelse(entity_position <= k_val, 1, 0)
      ) %>%
      arrange(fold) %>%
      select(fold, entity_position, MRR, Hit_k, k_1, k_3, k_10, k_20, k_50, k_100) %>%
      mutate(across(starts_with("k_"), ~round(.x, 4)))
    
    datatable(triple_rows, options = list(pageLength = 5, scrollX = TRUE), 
              colnames = c("Fold", "Rank", "MRR", paste0("Hits@", k_val), "Log-1", "Log-3", "Log-10", "Log-20", "Log-50", "Log-100"))
  })
  
  # --- Tuple CI Plot 1: LOG-K SCORES ---
  output$tuple_ci_plot_logk <- renderPlot({
    req(input$selected_triple_idx, best_pos())
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    eps <- sqrt(1/0.05)
    
    tuple_data <- best_pos() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      select(fold, k_1, k_3, k_10, k_20, k_50, k_100)
    
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
    req(input$selected_triple_idx, best_pos())
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    eps <- sqrt(1/0.05)
    
    tuple_data <- best_pos() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(
        MRR = 1 / entity_position,
        `Hits@1` = as.numeric(entity_position <= 1),
        `Hits@3` = as.numeric(entity_position <= 3),
        `Hits@10` = as.numeric(entity_position <= 10),
        `Hits@20` = as.numeric(entity_position <= 20),
        `Hits@50` = as.numeric(entity_position <= 50),
        `Hits@100` = as.numeric(entity_position <= 100)
      )
    
    n_folds <- nrow(tuple_data)
    
    plot_df <- data.frame(
      Metric = factor(c("Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR"), 
                      levels = c("Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR")),
      Mean = c(mean(tuple_data$`Hits@1`), mean(tuple_data$`Hits@3`), mean(tuple_data$`Hits@10`), 
               mean(tuple_data$`Hits@20`), mean(tuple_data$`Hits@50`), mean(tuple_data$`Hits@100`), 
               mean(tuple_data$MRR)),
      SD = c(sd(tuple_data$`Hits@1`), sd(tuple_data$`Hits@3`), sd(tuple_data$`Hits@10`), 
             sd(tuple_data$`Hits@20`), sd(tuple_data$`Hits@50`), sd(tuple_data$`Hits@100`), 
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
    req(input$selected_triple_idx, best_pos(), input$fold_for_top100)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    fold_sel <- as.integer(input$fold_for_top100)
    
    row <- best_pos() %>% filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail, fold == fold_sel)
    if (nrow(row) == 0) return(datatable(tibble(note = "No data"), options = list(dom = 't')))
    
    ents <- unlist(row$top_100_entities[[1]])
    scs  <- unlist(row$top_100_scores[[1]])
    tbl <- tibble(pos = seq_along(ents), entity = ents, score = round(scs, 4))
    datatable(tbl, options = list(pageLength = 5, scrollX = TRUE))
  })
  
  output$download_combined_csv <- downloadHandler(
    filename = function() paste0("metrics_", input$dataset, "_", input$model, "_", Sys.Date(), ".csv"),
    content = function(file) { write.csv(res_matrix(), file, row.names = FALSE) }
  )
}

shinyApp(ui = ui, server = server)