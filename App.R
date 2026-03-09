
# To Hit or Not To Hit: A Demo
# Authors: 
# Donatella Firmani
# Lorenzo Balzotti 
# Gian Mario Sangiovanni,
# Lucia Gallucci
# Giovanna Jona Lasinio

# Brief Description: 


# install.packages(c("bslib","shinyWidgets"))
# install.packages("shinycssloaders")

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
generate_dataset_with_folds <- function(dataset_name,
                                        folds = 15,
                                        seed = NULL) {
  
  # Returns a tibble with one row per (fold, triple) containing:
  # head, relation, tail, fold, position, score (logit of true tail),
  # softmax_true, top100_scores (numeric vector), top100_entities (char vector),
  # top100_cumexp (numeric cumexp vector length 100), sum_exp_all, n_entities
  
  if (!is.null(seed)) set.seed(seed)
  dati <- read.table(dataset_name, col.names = c("head", "relation", "tail"), 
                     blank.lines.skip = FALSE)[1:300, ]
  # entities <- paste0(entity_prefix, seq_len(n_entities)) # for now generate fake entities (why?)
  
  # create base test triples (same across folds)
  # heads <- paste0("h_", sample(letters, n_triples, replace = TRUE), sample(1000:9999, n_triples, replace = TRUE))
  # relations <- paste0("r_", sample(1:200, n_triples, replace = TRUE))
  # tails_idx <- sample(seq_len(n_entities), n_triples, replace = TRUE)
  # tails <- entities[tails_idx]
  
  heads <- as.character(dati$head)
  relations <- as.character(dati$relation)
  tails <- as.character(dati$tail)
  entities <- unique(c(dati$head, dati$tail))
  n_entities <- length(entities)
  entity_map <- setNames(seq_along(entities), entities)
  tails_idx <- entity_map[tails]
  
  # base_triples <- tibble(head = heads, relation = relations, tail = tails, tail_idx = tails_idx)
  base_triples <- tibble(head = heads, relation = relations, tail = tails, tails_idx = tails_idx)
  # For each fold we simulate logits ensuring the true tail has the specified position and top100 coherence
  
  n_rows <- folds * nrow(base_triples)
  rows <- vector("list", n_rows)
  
  idx <- 1
  for (f in seq_len(folds)) {
    for (i in seq_len(nrow(base_triples))) {
      bt <- base_triples[i, ]
      # position rule: with prob 0.5 <=20 else random in [21, n_entities]
      if (runif(1) < 0.5) {
        pos <- sample(seq_len(min(20, n_entities)), 1)
      } else {
        if (n_entities >= 21) {
          pos <- sample(21:n_entities, 1)
        } else {
          pos <- sample(seq_len(n_entities), 1) # security check useles
        }
      }
     
      
      # generate sorted scores (descending) and corresponding exps
      scores_sorted <- sort(rnorm(n_entities, mean = 0, sd = 1), decreasing = TRUE)
      #exps_sorted <- exp(scores_sorted - max(scores_sorted)) * exp(max(scores_sorted)) # stable but equivalent to exp(scores_sorted)
      
      # exps_sorted may be huge; we will compute sum_exp_all in stable manner
      # but since we use differences in numerator/denominator we can use exp(scores - max) approach:
      # choose denom_norm = sum(exp(scores_sorted - max_score)); then sum_exp_all = denom_norm * exp(max_score)
      
      max_score <- max(scores_sorted)
      exps_norm <- exp(scores_sorted - max_score)
      sum_exps_norm <- sum(exps_norm)
      #sum_exp_all <- sum_exps_norm * exp(max_score)  # keep this; but we'll use normalized approach below to avoid overflow
      
      # Build a random permutation of entity indices to assign ranks, but force the true tail to be at index 'pos'
      
      perm <- sample(seq_len(n_entities))
      true_idx <- as.integer(bt$tails_idx)
      
      # ensure perm[pos] == true_idx by swapping elements
      
      if (perm[pos] != true_idx) {
        kpos <- which(perm == true_idx)
        tmp <- perm[pos]
        perm[pos] <- true_idx
        perm[kpos] <- tmp
      }
      
      # create logits vector where logits[perm[i]] = scores_sorted[i]
      logits <- numeric(n_entities)
      logits[perm] <- scores_sorted
      
      # compute softmax (stable)
      logits_stable <- logits - max(logits)
      exps <- exp(logits_stable)
      sum_exps <- sum(exps)
      probs <- exps / sum_exps
      softmax_true <- probs[true_idx]
      score_true <- logits[true_idx]
      
      # prepare top100 lists
      topN <- min(100L, n_entities)
      top_indices_by_score <- perm[1:topN] # entities at the top positions
      top100_entities <- entities[top_indices_by_score]
      top100_scores <- scores_sorted[1:topN] # corresponding sorted scores
      
      # compute cumulative exp of top100 (with same stable normalization)
      top100_exps_norm <- exps_norm[1:topN] # note exps_norm corresponds to scores_sorted order
      top100_cumexp_norm <- cumsum(top100_exps_norm)
      
      # sum_exps_norm is sum(exp(scores - max_score)), but we used logits_stable to compute probs; however ratio of top100_cumexp_norm/sum_exps_norm equals mass_topk.
      # store metadata
      rows[[idx]] <- list(
        dataset = dataset_name,
        fold = f,
        head = bt$head,
        relation = bt$relation,
        tail = bt$tail,
        tail_idx = true_idx,
        position = pos,
        score = score_true,
        softmax_true = softmax_true,
        top100_scores = top100_scores,               # numeric vector length topN
        top100_entities = top100_entities,           # char vector length topN
        top100_cumexp_norm = top100_cumexp_norm,     # numeric cumulative exps normalized by max_score
        sum_exps_norm = sum_exps_norm,               # scalar (for normalization to compute mass_topk)
        n_entities = n_entities
      )
      idx = idx + 1
    }
  }
  
  # convert to tibble, keeping list-columns
  df <- tibble(
    dataset = map_chr(rows, "dataset"),
    fold = map_int(rows, "fold"),
    head = map_chr(rows, "head"),
    relation = map_chr(rows, "relation"),
    tail = map_chr(rows, "tail"),
    tail_idx = map_int(rows, "tail_idx"),
    position = map_int(rows, "position"),
    score = map_dbl(rows, "score"),
    softmax_true = map_dbl(rows, "softmax_true"),
    top100_scores = map(rows, "top100_scores"),
    top100_entities = map(rows, "top100_entities"),
    top100_cumexp_norm = map(rows, "top100_cumexp_norm"),
    sum_exps_norm = map_dbl(rows, "sum_exps_norm"),
    n_entities = map_int(rows, "n_entities")
  )
  
  df
}

# ---------- Build all three datasets (simulate) -------------
# NOTE: choose sizes moderate so interactive app is responsive. Adjust as desired.


generate_all_datasets <- function(seed = 2026) {
  set.seed(seed)
  list(
    "YAGO3-10" = generate_dataset_with_folds("train_YAGO3-10.txt", folds = 15, seed = seed + 1),
    "WN18RR"  = generate_dataset_with_folds("train_WN18RR.txt", folds = 15,  seed = seed + 2),
    "FB15k-237" = generate_dataset_with_folds("train_FB15k-237.txt", folds = 15, seed = seed + 3)
  )
}

# Precompute datasets once (in memory). If you prefer heavy generation per run, change this.
ALL_DATASETS <- generate_all_datasets(seed = 12345)

# ---------- Shiny UI ------------

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
    
    body {
      background-color: #F7F9FC;
    }

    .title-panel {
      background: linear-gradient(90deg,#1F4E79,#2E86C1);
      color: white;
      padding: 20px;
      border-radius: 8px;
      margin-bottom: 20px;
    }

    .card {
      border-radius: 10px;
      box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
    }

    .control-label {
      font-weight: 600;
    }

    .plot-title {
      font-size:18px;
      font-weight:600;
    }

    "))
  ),
  
  div(class="title-panel",
      h2("2HN2H — Evaluating Knowledge Graph Completion"),
      p("Interactive demo: Hits@k vs Top-k Log Score across datasets and folds")
  ),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 3,
      
      h4("Configuration"),
      
      selectInput(
        "dataset",
        "Dataset",
        choices = c("YAGO3-10","WN18RR","FB15k-237")
      ),
      
      sliderInput(
        "k",
        "Top-k value",
        min = 1,
        max = 100,
        value = 10
      ),
      
      sliderInput(
        "n_sample_triples",
        "Triples to inspect",
        min = 5,
        max = 80,
        value = 20
      ),
      
      numericInput(
        "seed",
        "Random seed",
        value = 42
      ),
      
      br(),
      
      actionBttn(
        "resample",
        "Resample Triples",
        style = "gradient",
        color = "primary"
      ),
      
      br(),br(),
      
      checkboxInput(
        "show_all_folds_metrics",
        "Show individual folds"
      ),
      
      downloadButton(
        "download_combined_csv",
        "Download Dataset"
      )
      
    ),
    
    mainPanel(
      
      tabsetPanel(
        id="tabs",
        
        tabPanel(
          "Model Evaluation",
          
          br(),
          
          fluidRow(
            
            column(
              6,
              card(
                card_header("Hits@k Performance"),
                card_body(
                  shinycssloaders::withSpinner(plotOutput("hits_barplot", height=350))
                )
              )
            ),
            
            column(
              6,
              card(
                card_header("Top-k Log Score"),
                card_body(
                  shinycssloaders::withSpinner(plotOutput("logk_barplot", height=350))
                )
              )
            )
            
          ),
          
          br(),
          
          fluidRow(
            
            column(
              6,
              card(
                card_header("Prediction Stability Across Folds"),
                card_body(
                  shinycssloaders::withSpinner(plotOutput("stability_plot", height=350))
                )
              )
            ),
            
            column(
              6,
              card(
                card_header("Rank vs Log-Score"),
                card_body(
                  shinycssloaders::withSpinner(plotOutput("position_vs_logk", height=350))
                )
              )
            )
            
          )
          
        ),
        
        tabPanel(
          "Triple Inspector",
          
          br(),
          
          fluidRow(
            
            column(
              6,
              card(
                card_header("Sampled Triples Summary"),
                card_body(
                  DTOutput("triples_table_summary")
                )
              )
            ),
            
            column(
              6,
              card(
                card_header("Prediction Details Across Folds"),
                card_body(
                  
                  uiOutput("selected_triple_ui"),
                  
                  br(),
                  
                  DTOutput("triple_fold_table"),
                  
                  br(),
                  
                  selectInput(
                    "fold_for_top100",
                    "Fold for Top-100 Predictions",
                    choices = as.character(1:15)
                  ),
                  
                  DTOutput("top100_table")
                  
                )
              )
            )
            
          )
          
        ),
        
        tabPanel(
          "Dataset Preview",
          
          br(),
          
          card(
            card_header("Raw Data Preview"),
            card_body(
              DTOutput("raw_preview")
            )
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
  
  # Reactive: get current dataset tibble
  ds_all <- reactive({
    ALL_DATASETS[[input$dataset]]
  })
  
  # Compute per-fold aggregated metrics (Hits@k and mean top-k log score)
  per_fold_metrics <- reactive({
    df <- ds_all()
    k <- input$k
    # for each row compute logk using stored normalized exps:
    df2 <- df %>%
      mutate(
        # mass_topk = top100_cumexp_norm[k] / sum_exps_norm  (only valid for k<=100)
        mass_topk = map2_dbl(top100_cumexp_norm, sum_exps_norm, ~ {
          cumvec <- .x
          denom <- .y
          kk <- min(k, length(cumvec))
          cumvec[kk] / denom
        }),
        logk = pmap_dbl(list(position, softmax_true, mass_topk, n_entities),
                        function(position, softmax_true, mass_topk, n_entities) {
                          if (position <= k) {
                            -log(max(softmax_true, 1e-300))
                          } else {
                            pi_k <- (1 - mass_topk) / (n_entities - k)
                            -log(max(pi_k, 1e-300))
                          }
                        }),
        hit_k = as.integer(position <= k)
      )
    # now aggregate per fold
    per_fold <- df2 %>%
      group_by(fold) %>%
      summarise(
        hits_at_k = mean(hit_k),
        mean_logk = mean(logk),
        sd_logk = sd(logk),
        n_triples = n(),
        .groups = "drop"
      ) %>% arrange(fold)
    list(per_fold = per_fold, df_with_logk = df2)
  })
  
  # Plot: Hits@k across folds (bars) and mean ± 95% CI
  output$hits_barplot <- renderPlot({
    res <- per_fold_metrics()
    per_fold <- res$per_fold
    if (input$show_all_folds_metrics) {
      ggplot(per_fold, aes(x = factor(fold), y = hits_at_k)) +
        geom_col(fill = hits_color) +
        geom_hline(aes(yintercept = mean(hits_at_k)), color = "red", linetype = "dashed") +
        labs(title = paste0("Hits@", input$k, " per fold (red dashed = mean)"),
             x = "Fold", y = paste0("Hits@", input$k)) +
        ylim(0,1) +
        theme_minimal()
    } else {
      # compute mean & 95% CI across folds
      mean_hits <- mean(per_fold$hits_at_k)
      se <- sd(per_fold$hits_at_k) / sqrt(nrow(per_fold))
      ci_low <- mean_hits - 1.96 * se
      ci_high <- mean_hits + 1.96 * se
      summary_df <- tibble(metric = "Hits@k", mean = mean_hits, ci_low = ci_low, ci_high = ci_high)
      ggplot(summary_df, aes(x = metric, y = mean)) +
        geom_col(width = 0.4, fill = hits_color) +
        geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.12) +
        geom_text(aes(label = sprintf("%.3f", mean)), vjust = -1.2) +
        ylim(0,1) +
        labs(title = paste0("Mean Hits@", input$k, " across folds (95% CI)"), x = NULL, y = "") +
        theme_minimal()
    }
  })
  
  # Plot: top-k log score mean ± 95% CI across folds (lower is better)
  output$logk_barplot <- renderPlot({
    res <- per_fold_metrics()
    per_fold <- res$per_fold
    if (input$show_all_folds_metrics) {
      ggplot(per_fold, aes(x = factor(fold), y = mean_logk)) +
        geom_col(fill = hits_color) +
        geom_hline(aes(yintercept = mean(mean_logk)), color = "red", linetype = "dashed") +
        labs(title = paste0("Mean top-k log score per fold (k=", input$k, ")"),
             x = "Fold", y = "mean top-k log score (-log prob)") +
        theme_minimal()
    } else {
      mean_logk <- mean(per_fold$mean_logk)
      se <- sd(per_fold$mean_logk) / sqrt(nrow(per_fold))
      ci_low <- mean_logk - 1.96 * se
      ci_high <- mean_logk + 1.96 * se
      summary_df <- tibble(metric = "top-k log score", mean = mean_logk, ci_low = ci_low, ci_high = ci_high)
      ggplot(summary_df, aes(x = metric, y = mean)) +
        geom_col(width = 0.4, fill = "tomato") +
        geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.12) +
        geom_text(aes(label = sprintf("%.3f", mean)), vjust = -1.2) +
        labs(title = paste0("Mean top-k log score across folds (k=", input$k, ") — lower better"), x = NULL, y = "") +
        theme_minimal()
    }
  })
  
  # Stability plot: for each unique triple compute mean_logk and sd_logk across folds
  output$stability_plot <- renderPlot({
    df2 <- per_fold_metrics()$df_with_logk
    stat <- df2 %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_logk = mean(logk),
        sd_logk = sd(logk),
        mean_position = mean(position),
        proportion_hits = mean(position <= input$k),
        .groups = "drop"
      )
    ggplot(stat, aes(x = mean_logk, y = sd_logk)) +
      geom_point(alpha = 0.5) +
      labs(title = "Stability across folds: mean top-k log score vs sd (per triple)",
           x = "Mean top-k log score (lower better)", y = "SD of top-k log score across folds") +
      theme_minimal()
  })
  
  # Position vs logk per triple (aggregate)
  output$position_vs_logk <- renderPlot({
    df2 <- per_fold_metrics()$df_with_logk
    stat <- df2 %>%
      group_by(head, relation, tail) %>%
      summarise(mean_position = mean(position), mean_logk = mean(logk), .groups = "drop")
    ggplot(stat, aes(x = mean_position, y = mean_logk)) +
      geom_jitter(alpha = 0.4, height = 0, width = 0.6) +
      labs(title = "Mean position vs mean top-k log score (per triple)",
           x = "Mean position (lower = better rank)", y = "Mean top-k log score (-log prob)") +
      theme_minimal()
  })
  
  # Reactive: sampled distinct triples for inspector (resample button)
  sampled_triples <- eventReactive(input$resample, {
    set.seed(input$seed)
    df2 <- per_fold_metrics()$df_with_logk
    unique_triples <- df2 %>% distinct(head, relation, tail)
    n <- min(nrow(unique_triples), input$n_sample_triples)
    unique_triples %>% slice_sample(n = n)
  }, ignoreNULL = FALSE)
  
  # Triples summary table: mean position, mean logk, sd logk across folds, proportion hits
  output$triples_table_summary <- renderDT({
    sampled <- sampled_triples()
    df2 <- per_fold_metrics()$df_with_logk
    stat <- df2 %>%
      group_by(head, relation, tail) %>%
      summarise(
        mean_position = mean(position),
        mean_logk = mean(logk),
        sd_logk = sd(logk),
        prop_hits = mean(position <= input$k),
        .groups = "drop"
      ) %>%
      inner_join(sampled, by = c("head", "relation", "tail")) %>%
      arrange(mean_logk)
    datatable(stat, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # UI for selecting one triple from sampled ones
  output$selected_triple_ui <- renderUI({
    sampled <- sampled_triples()
    choices <- paste0(seq_len(nrow(sampled)), ": ", sampled$head, " | ", sampled$relation, " | ", sampled$tail)
    selectInput("selected_triple_idx", "Choose triple (from sample)", choices = choices, selected = choices[1])
  })
  
  # Table: per-fold metrics (position, logk) for selected triple
  output$triple_fold_table <- renderDT({
    req(input$selected_triple_idx)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    sampled <- sampled_triples()
    chosen <- sampled[sel_idx, ]
    df2 <- per_fold_metrics()$df_with_logk
    triple_rows <- df2 %>% filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail) %>%
      mutate(logk = round(logk, 4), softmax_true = signif(softmax_true, 6)) %>%
      arrange(fold) %>%
      select(fold, position, score, softmax_true, logk)
    datatable(triple_rows, options = list(pageLength = 15, scrollX = TRUE), colnames = c("Fold","Position","Score (logit)","softmax(tail)","top-k log score"))
  })
  
  # Top100 table for the chosen triple & fold
  output$top100_table <- renderDT({
    req(input$selected_triple_idx)
    sel_idx <- as.integer(strsplit(input$selected_triple_idx, ":")[[1]][1])
    sampled <- sampled_triples()
    chosen <- sampled[sel_idx, ]
    df2 <- per_fold_metrics()$df_with_logk
    fold_sel <- as.integer(input$fold_for_top100)
    row <- df2 %>% filter(head == chosen$head, relation == chosen$relation, tail == chosen$tail, fold == fold_sel)
    if (nrow(row) == 0) {
      return(datatable(tibble(note = "No data for this fold/triple"), options = list(dom = 't')))
    }
    ents <- unlist(row$top100_entities[[1]])
    scs  <- unlist(row$top100_scores[[1]])
    tbl <- tibble(pos = seq_along(ents), entity = ents, score = round(scs, 4))
    datatable(tbl, options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Raw preview (combined)
  output$raw_preview <- renderDT({
    df2 <- per_fold_metrics()$df_with_logk
    preview <- df2 %>%
      group_by(fold) %>%
      slice_head(n = 10) %>%
      ungroup() %>%
      select(fold, head, relation, tail, position, score, softmax_true)
    datatable(preview, options = list(pageLength = 15, scrollX = TRUE))
  })
  
  # Download combined CSV: flatten top100 vectors to pipe-separated strings
  output$download_combined_csv <- downloadHandler(
    filename = function() paste0("simulated_", input$dataset, "_folds_combined_", Sys.Date(), ".csv"),
    content = function(file) {
      df_all <- per_fold_metrics()$df_with_logk
      out <- df_all %>%
        mutate(
          top100_scores = map_chr(top100_scores, ~ paste(round(.x, 6), collapse = "|")),
          top100_entities = map_chr(top100_entities, ~ paste(.x, collapse = "|")),
          top100_cumexp_norm = map_chr(top100_cumexp_norm, ~ paste(round(.x, 8), collapse = "|"))
        ) %>%
        select(dataset, fold, head, relation, tail, position, score, softmax_true, n_entities,
               top100_scores, top100_entities, top100_cumexp_norm)
      write.csv(out, file, row.names = FALSE)
    }
  )
  
}

# ---------- Run app ----------
shinyApp(ui = ui, server = server)

