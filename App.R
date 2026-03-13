# To Hit or Not To Hit: A Demo
# Authors:
# Marco Lattanzi
# Johannes Resin
# Donatella Firmani
# Lorenzo Balzotti 
# Gian Mario Sangiovanni,
# Lucia Gallucci
# Giovanna Jona Lasinio

# Brief Description: 


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
      h2("2HN2H — Evaluating Knowledge Graph Completion"),
      p("Interactive demo: Hits@k, MRR & Log Score stratified by Dataset & Model")
  ),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 3,
      h4("Configuration"),
      
      selectInput("dataset", "Dataset", choices = c("YAGO3_10")),
      selectInput("model", "Model", choices = c("RotatE")),
      selectInput("k", "Top-k value (for Hits & LogK)", choices = c("1", "3", "10", "20", "50", "100"), selected = "10"),
      
      sliderInput("n_sample_triples", "Triples to inspect", min = 5, max = 80, value = 20),
      numericInput("seed", "Random seed", value = 42),
      
      br(),
      actionBttn("resample", "Resample Triples", style = "gradient", color = "primary"),
      helpText("Only samples triples where position < 100 in at least 50% of folds."),
      br(),br(),
      
      checkboxInput("show_all_folds_metrics", "Show individual folds", value = TRUE),
      downloadButton("download_combined_csv", "Download Current View")
    ),
    
    mainPanel(
      tabsetPanel(
        id="tabs",
        tabPanel("Model Evaluation",
                 br(),
                 fluidRow(
                   column(6, card(card_header("Hits@k Performance"), card_body(shinycssloaders::withSpinner(plotOutput("hits_forest", height=350))))),
                   column(6, card(card_header("Mean Reciprocal Rank (MRR)"), card_body(shinycssloaders::withSpinner(plotOutput("mrr_forest", height=350)))))
                 ),
                 br(),
                 fluidRow(
                   column(6, card(card_header("Top-k Log Score (Softmax)"), card_body(shinycssloaders::withSpinner(plotOutput("logk_forest", height=350))))),
                   column(6, card(card_header("Rank vs Softmax Score"), card_body(shinycssloaders::withSpinner(plotOutput("position_vs_softmax", height=350)))))
                 )
        ),
        tabPanel("Triple Inspector",
                 br(),
                 fluidRow(
                   column(6, card(card_header("Sampled Triples Summary"), card_body(DTOutput("triples_table_summary")))),
                   column(6, card(card_header("Prediction Details Across Folds"), card_body(
                     uiOutput("selected_triple_ui"), br(),
                     DTOutput("triple_fold_table"), br(),
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
        plot.title = element_text(face="bold", size=16),
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
  
  # --- Forest Plot: Hits@k ---
  output$hits_forest <- renderPlot({
    req(res_matrix())
    mat <- res_matrix()
    k_val <- input$k
    mean_col <- paste0("hits@", k_val, "_mean")
    low_col <- paste0("hits@", k_val, "_ci_low")
    high_col <- paste0("hits@", k_val, "_ci_high")
    
    if (input$show_all_folds_metrics) {
      ggplot(mat, aes(x = factor(fold), y = .data[[mean_col]])) +
        geom_point(color = "#1F4E79", size = 3) +
        geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[high_col]]), width = 0.2, color = "#1F4E79") +
        coord_flip() +
        geom_hline(aes(yintercept = mean(.data[[mean_col]])), color = "red", linetype = "dashed") +
        labs(title = paste0("Hits@", k_val, " per fold"), x = "Fold", y = paste0("Hits@", k_val))
    } else {
      overall_mean <- mean(mat[[mean_col]], na.rm=TRUE)
      ggplot(tibble(x="Overall", y=overall_mean), aes(x=x, y=y)) +
        geom_point(color = "#1F4E79", size = 4) +
        coord_flip() + ylim(0, 1) +
        labs(title = paste0("Overall Mean Hits@", k_val), x = NULL, y = "")
    }
  })
  
  # --- Forest Plot: MRR ---
  output$mrr_forest <- renderPlot({
    req(res_matrix())
    mat <- res_matrix()
    
    if (input$show_all_folds_metrics) {
      ggplot(mat, aes(x = factor(fold), y = mrr_mean)) +
        geom_point(color = "#27AE60", size = 3) +
        geom_errorbar(aes(ymin = mrr_ci_low, ymax = mrr_ci_high), width = 0.2, color = "#27AE60") +
        coord_flip() +
        geom_hline(aes(yintercept = mean(mrr_mean)), color = "red", linetype = "dashed") +
        labs(title = "MRR per fold", x = "Fold", y = "MRR")
    } else {
      overall_mean <- mean(mat$mrr_mean, na.rm=TRUE)
      ggplot(tibble(x="Overall", y=overall_mean), aes(x=x, y=y)) +
        geom_point(color = "#27AE60", size = 4) +
        coord_flip() + ylim(0, 1) +
        labs(title = "Overall Mean MRR", x = NULL, y = "")
    }
  })
  
  # --- Forest Plot: LogK Softmax ---
  output$logk_forest <- renderPlot({
    req(res_matrix())
    mat <- res_matrix()
    k_val <- input$k
    mean_col <- paste0("log_", k_val, "_softmax_mean")
    low_col <- paste0("log_", k_val, "_softmax_ci_low")
    high_col <- paste0("log_", k_val, "_softmax_ci_high")
    
    if (input$show_all_folds_metrics) {
      ggplot(mat, aes(x = factor(fold), y = .data[[mean_col]])) +
        geom_point(color = "#E67E22", size = 3) +
        geom_errorbar(aes(ymin = .data[[low_col]], ymax = .data[[high_col]]), width = 0.2, color = "#E67E22") +
        coord_flip() +
        geom_hline(aes(yintercept = mean(.data[[mean_col]], na.rm=TRUE)), color = "red", linetype = "dashed") +
        labs(title = paste0("Mean top-", k_val, " log score (Softmax)"), x = "Fold", y = "Log Score")
    } else {
      overall_mean <- mean(mat[[mean_col]], na.rm=TRUE)
      ggplot(tibble(x="Overall", y=overall_mean), aes(x=x, y=y)) +
        geom_point(color = "#E67E22", size = 4) +
        coord_flip() +
        labs(title = paste0("Overall Mean top-", k_val, " log score"), x = NULL, y = "")
    }
  })
  
  # --- Position vs Softmax (replacing logk stability) ---
  output$position_vs_softmax <- renderPlot({
    req(best_pos())
    bp <- best_pos()
    
    stat <- bp %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_position = mean(entity_position, na.rm=TRUE), 
        mean_softmax = mean(softmax_score, na.rm=TRUE), 
        .groups = "drop"
      )
    
    ggplot(stat, aes(x = mean_position, y = mean_softmax)) +
      geom_jitter(alpha = 0.4, height = 0, width = 0.6, color="#8E44AD") +
      labs(title = "Mean rank vs Mean Softmax Score", x = "Mean position (rank)", y = "Mean Softmax Score")
  })
  
  # --- Resampling Filtered Triples ---
  sampled_triples <- eventReactive(input$resample, {
    req(best_pos())
    set.seed(input$seed)
    
    # FILTER: Only keep triples where entity_position < 100 in at least 50% of the folds
    valid_triples <- best_pos() %>%
      group_by(head, relation, tail) %>%
      summarise(prop_under_100 = mean(entity_position < 100, na.rm = TRUE), .groups = "drop") %>%
      filter(prop_under_100 >= 0.5)
    
    n <- min(nrow(valid_triples), input$n_sample_triples)
    valid_triples %>% slice_sample(n = n)
  }, ignoreNULL = FALSE)
  
  # --- Triples summary table (No logk, added MRR) ---
  output$triples_table_summary <- renderDT({
    req(best_pos(), sampled_triples())
    
    stat <- best_pos() %>%
      mutate(mrr = 1 / entity_position) %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_pos = round(mean(entity_position, na.rm=TRUE), 2),
        mean_mrr = round(mean(mrr, na.rm=TRUE), 4),
        mean_softmax = signif(mean(softmax_score, na.rm=TRUE), 4),
        .groups = "drop"
      ) %>%
      inner_join(sampled_triples(), by = c("head", "relation", "tail")) %>%
      arrange(mean_pos) # Sorted by best position
    
    datatable(stat %>% select(-prop_under_100), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # --- UI for selecting a triple ---
  output$selected_triple_ui <- renderUI({
    req(sampled_triples())
    sampled <- sampled_triples()
    choices <- paste0(seq_len(nrow(sampled)), ": ", sampled$head, " | ", sampled$relation, " | ", sampled$tail)
    selectInput("selected_triple_idx", "Choose triple (from sample)", choices = choices, selected = choices[1])
  })
  
  # --- Fold Details Table ---
  output$triple_fold_table <- renderDT({
    req(input$selected_triple_idx, best_pos())
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    chosen <- sampled_triples()[sel_idx, ]
    
    triple_rows <- best_pos() %>% 
      filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(mrr = round(1 / entity_position, 4), softmax = signif(softmax_score, 6)) %>%
      arrange(fold) %>%
      select(fold, entity_position, mrr, softmax)
    
    datatable(triple_rows, options = list(pageLength = 15, scrollX = TRUE), 
              colnames = c("Fold","Position (Rank)","MRR", "Softmax(tail)"))
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
    datatable(tbl, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$download_combined_csv <- downloadHandler(
    filename = function() paste0("metrics_", input$dataset, "_", input$model, "_", Sys.Date(), ".csv"),
    content = function(file) { write.csv(res_matrix(), file, row.names = FALSE) }
  )
}

shinyApp(ui = ui, server = server)