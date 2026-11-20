
# To Hit or Not To Hit: A Demo

# Authors:
# Gian Mario Sangiovanni, Lucia Gallucci, Lorenzo Balzotti, Donatella Firmani, Giovanna Jona Lasinio, Johannes Resin


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

# Settings ---------------------------------------------------------------------

K_RANGE          <- c(1, 3, 10, 20)            # top-k values used in the whole app
CHEB_EPS         <- sqrt(1 / 0.05)             # Chebyshev: P(|X - mu| >= eps * sd) <= 1 / eps^2 = 0.05
BUILTIN_DATASETS <- c("YAGO3_10", "WN18RR", "KINSHIPS", "FB15k237")
BUILTIN_MODELS   <- c("RotatE", "HolE", "MurE", "ComplEx")
USER_DATASET     <- "Your data"                # entry of the Dataset menu for the uploaded data
BYOD_MIN_OVERLAP <- 0.9                        # min share of test triples present in every (model, fold) table
BYOD_EXAMPLE     <- "example_YAGO3_10.zip"     # example files offered in the BYOD tab (if the file is in the app folder)
options(shiny.maxRequestSize = 300 * 1024^2)   # uploads up to 300 MB (Shiny's default is 5 MB)

# Metrics ----------------------------------------------------------------------

# Probabilistic top-k scores.
# The predictive distribution of a model over the |E| candidate tails is the softmax
# of its scores: p(e) = exp(s(h, r, e)) / sum_exp. Its top-k version keeps the k most
# likely entities and spreads the remaining mass 1 - M_k uniformly over the other
# |E| - k entities, so that the true tail t gets
#   q(t) = p(t)                     if rank(t) <= k
#   q(t) = (1 - M_k) / (|E| - k)    otherwise
# Top-k log score:   -log q(t)
# Top-k Brier score: sum_e (q(e) - 1{e = t})^2
# Inputs (one value per test triple): rank, score_true, log_z = log(sum_exp);
# top = matrix of the highest scores, one row per triple, in decreasing order.
compute_topk_scores <- function(rank, score_true, log_z, top, n_entities, k_range = K_RANGE) {
  p_top  <- exp(top[, seq_len(max(k_range)), drop = FALSE] - log_z)  # probabilities of the top entities
  p_true <- exp(score_true - log_z)                                   # probability of the true tail
  out <- tibble(softmax_score = p_true)
  for (k in k_range) {
    m_k  <- rowSums(p_top[, seq_len(k), drop = FALSE])                # mass of the top-k list
    rest <- pmax(1 - m_k, 0) / (n_entities - k)                       # probability of each entity outside the list
    q_t  <- ifelse(rank <= k, p_true, rest)                           # probability of the true tail
    out[[paste0("k_", k)]]          <- -log(pmax(q_t, 1e-300))
    out[[paste0("orig_brier_", k)]] <- rowSums(p_top[, seq_len(k), drop = FALSE]^2) +
                                       (n_entities - k) * rest^2 - 2 * q_t + 1
  }
  out
}

# Mean and Chebyshev interval of every metric over the test triples of each fold
summarise_by_fold <- function(per_triple, eps = CHEB_EPS) {
  for (k in K_RANGE) per_triple[[paste0("hits@", k)]] <- as.numeric(per_triple$entity_position <= k)
  metrics <- c(paste0("hits@", K_RANGE), "mrr", paste0("orig_brier_", K_RANGE), paste0("k_", K_RANGE))
  per_triple %>%
    group_by(fold) %>%
    summarise(
      across(
        all_of(metrics),
        list(
          mean    = ~mean(.x, na.rm = TRUE),
          ci_low  = ~mean(.x, na.rm = TRUE) - eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x)))),
          ci_high = ~mean(.x, na.rm = TRUE) + eps * (sd(.x, na.rm = TRUE) / sqrt(sum(!is.na(.x))))
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    )
}

fmt_int <- function(x) formatC(x, format = "d", big.mark = ",")
  
# Built-in datasets ------------------------------------------------------------

# combined_<dataset>_<model>.parquet: one row per (fold, test triple), from our pipeline.
# Hits@k and MRR come from the rank (column `hits`). The Top-k log and Brier scores are the
# pipeline's columns log_k_softmax and orig_brier_k, computed with the number of entities |E|
# of each fold; for uploaded data compute_topk_scores() gives the same values.
read_dataset_with_folds <- function(dataset_name, model_name) {
  file_path <- paste0("combined_", dataset_name, "_", model_name, ".parquet")
  cols <- c("fold", "head", "relation", "tail", "hits", "softmax",
            paste0("log_", K_RANGE, "_softmax"), paste0("orig_brier_", K_RANGE))
  dati <- read_parquet(file_path, col_select = all_of(cols))
  best_position <- tibble(
    entity_position = dati$hits,              # rank of the true tail (1 = best)
    softmax_score   = dati$softmax,           # model probability of the true tail
    fold            = dati$fold,
    mrr             = 1 / dati$hits,
    head            = as.character(dati$head),
    relation        = as.character(dati$relation),
    tail            = as.character(dati$tail)
  )
  for (k in K_RANGE) {
    best_position[[paste0("k_", k)]]          <- dati[[paste0("log_", k, "_softmax")]]
    best_position[[paste0("orig_brier_", k)]] <- dati[[paste0("orig_brier_", k)]]
  }
  list(results_matrix = summarise_by_fold(best_position), best_position = best_position)
}

# Built-in results are read once and kept in memory (shared by all the sessions)
builtin_cache <- new.env(parent = emptyenv())
load_builtin <- function(dataset_name, model_name) {
  key <- paste(dataset_name, model_name, sep = "/")
  if (is.null(builtin_cache[[key]])) builtin_cache[[key]] <- read_dataset_with_folds(dataset_name, model_name)
  builtin_cache[[key]]
}

# Bring Your Own Data ----------------------------------------------------------

# Columns of the uploaded tables (matched by name, in any order; other columns are ignored)
BYOD_SPEC <- data.frame(
  column  = c("head", "relation", "tail", "rank", "score_true", "sum_exp", "top_scores"),
  type    = c("text", "text", "text", "integer\u00A0\u2265\u00A01", "number", "number\u00A0>\u00A00", "list of numbers"),
  content = c("head entity h",
              "relation r",
              "true tail entity t",
              "rank of t among all the candidate tails (1 = best)",
              "score s(h, r, t) of the true tail",
              "\u03A3 exp s(h, r, e) over all the entities e, i.e. the softmax normaliser (or log_sum_exp, its logarithm)",
              paste0("the highest scores s(h, r, e), in decreasing order: at least ", max(K_RANGE), " values (extra values are ignored)"))
)
BYOD_N_TOP   <- max(K_RANGE)   # top scores needed by the metrics
BYOD_ALIASES <- c(hits = "rank", score_right = "score_true", top_100_scores = "top_scores")  # names in our own files
BYOD_STATUS  <- list(ok = c("\u2714", "#1E8449"), warn = c("\u26A0", "#B9770E"), error = c("\u2716", "#C0392B"))

byod_spec_table <- function() {
  tags$table(
    class = "table table-sm",
    tags$thead(tags$tr(tags$th("Column"), tags$th("Type"), tags$th("Content"))),
    tags$tbody(lapply(seq_len(nrow(BYOD_SPEC)), function(i)
      tags$tr(tags$td(tags$code(BYOD_SPEC$column[i])), tags$td(BYOD_SPEC$type[i]), tags$td(BYOD_SPEC$content[i]))))
  )
}

# "RotatE_fold3.csv" -> model "RotatE", fold 3;  "RotatE.csv" -> model "RotatE", fold from the column `fold`
byod_parse_name <- function(file_name) {
  stem  <- tools::file_path_sans_ext(basename(file_name))
  parts <- regmatches(stem, regexec("^(.+?)[_. -]*fold[_. -]*([0-9]+)$", stem, ignore.case = TRUE))[[1]]
  if (length(parts) == 3) list(model = parts[2], fold = as.integer(parts[3]))
  else list(model = stem, fold = NA_integer_)
}

byod_read_table <- function(path, file_name) {
  ext <- tolower(tools::file_ext(file_name))
  if (ext == "parquet") return(as.data.frame(read_parquet(path)))
  if (ext == "csv") return(utils::read.csv(path, colClasses = "character", check.names = FALSE,
                                           na.strings = character(0), strip.white = TRUE,
                                           fileEncoding = "UTF-8-BOM"))   # also CSV saved by Excel
  stop("unsupported file type '.", ext, "': use .parquet or .csv")
}

as_num <- function(x) suppressWarnings(as.numeric(x))

# top_scores: list column (Parquet) or strings "1.2;0.8;..." (CSV; "[1.2, 0.8, ...]" works too)
byod_parse_scores <- function(x) {
  if (is.list(x)) return(lapply(x, as_num))
  x <- gsub("^\\s*\\[|\\]\\s*$", "", as.character(x))
  lapply(strsplit(x, "[;,[:space:]]+"), function(v) as_num(v[nzchar(v)]))
}

fmt_folds <- function(f) {
  f <- sort(unique(f))
  if (length(f) > 2 && all(diff(f) == 1)) paste0(f[1], "\u2013", f[length(f)]) else paste(f, collapse = ", ")
}

byod_file_error <- function(file_name, msg, model = "") {
  list(file = file_name, model = model, folds = "", data = NULL, top = NULL,
       errors = msg, warnings = character(0), notes = character(0))
}

# Reads and checks one uploaded table. Returns file, model, folds, messages (errors,
# warnings, notes) and, when there are no errors, the data needed by the metrics.
byod_read_file <- function(path, file_name) {
  info <- byod_parse_name(file_name)
  res  <- byod_file_error(file_name, character(0), info$model)
  fail <- function(msg) { res$errors <- msg; res }

  if (!nzchar(info$model)) return(fail("cannot read the model name from the file name (expected <model>_fold<f>)"))
  d <- tryCatch(byod_read_table(path, file_name), error = function(e) e)
  if (inherits(d, "error")) return(fail(paste("cannot read the file:", conditionMessage(d))))
  if (nrow(d) == 0) return(fail("the table is empty"))
  if (ncol(d) == 1 && grepl(";", names(d)))
    return(fail("only one column found: the file seems to use ';' as separator, use ',' (standard CSV)"))

  # column names: case-insensitive, plus the names used in our own files
  names(d) <- tolower(trimws(names(d)))
  for (a in names(BYOD_ALIASES))
    if (a %in% names(d) && !(BYOD_ALIASES[[a]] %in% names(d))) names(d)[names(d) == a] <- BYOD_ALIASES[[a]]

  need    <- c("head", "relation", "tail", "rank", "score_true", "top_scores", if (is.na(info$fold)) "fold")
  missing <- setdiff(need, names(d))
  if (!any(c("sum_exp", "log_sum_exp") %in% names(d))) missing <- c(missing, "sum_exp")
  if (length(missing))
    return(fail(paste0("missing column(s): ", paste(missing, collapse = ", "),
                       if ("fold" %in% missing) " (or name the file <model>_fold<f>)")))
  extra <- setdiff(names(d), c(need, "sum_exp", "log_sum_exp", "fold"))
  if (length(extra)) res$notes <- sprintf("%d extra column(s) ignored", length(extra))

  # types and values
  ent <- lapply(d[c("head", "relation", "tail")], function(v) trimws(as.character(v)))
  if (any(vapply(ent, function(v) any(is.na(v) | v == ""), logical(1))))
    return(fail("head, relation and tail cannot be empty"))
  rank <- as_num(d$rank)
  if (anyNA(rank) || any(rank < 1 | rank != round(rank))) return(fail("rank must contain integers \u2265 1"))
  score <- as_num(d$score_true)
  if (!all(is.finite(score))) return(fail("score_true must contain numbers"))
  log_z <- if ("log_sum_exp" %in% names(d)) as_num(d$log_sum_exp) else suppressWarnings(log(as_num(d$sum_exp)))
  if (!all(is.finite(log_z))) return(fail("sum_exp must contain positive numbers (log_sum_exp: finite numbers)"))
  fold <- if (is.na(info$fold)) as_num(d$fold) else rep(info$fold, nrow(d))
  if (anyNA(fold) || any(fold != round(fold))) return(fail("fold must contain integers"))

  # top scores: at least BYOD_N_TOP numbers per row, in decreasing order
  top_list <- byod_parse_scores(d$top_scores)
  short    <- lengths(top_list) < BYOD_N_TOP
  if (any(short))
    return(fail(sprintf("top_scores needs at least %d values per row (%d rows have fewer)", BYOD_N_TOP, sum(short))))
  top <- matrix(unlist(lapply(top_list, `[`, seq_len(BYOD_N_TOP))), ncol = BYOD_N_TOP, byrow = TRUE)
  if (!all(is.finite(top))) return(fail("top_scores must contain numbers"))
  if (any(top[, -1] > top[, -BYOD_N_TOP] + 1e-9))
    return(fail("top_scores must be sorted in decreasing order (if your model outputs distances, use score = -distance)"))

  # probabilities cannot exceed 1: sum_exp must be the sum over all the entities
  if (any(exp(score - log_z) > 1 + 1e-6) || any(rowSums(exp(top - log_z)) > 1 + 1e-6))
    return(fail("probabilities larger than 1: sum_exp must be the sum of exp(score) over all the entities (if you have its logarithm, name the column log_sum_exp)"))

  # each test triple once per fold
  key <- paste(fold, ent$head, ent$relation, ent$tail, sep = "\t")
  if (anyDuplicated(key)) return(fail(sprintf("%d duplicated test triple(s) in the same fold", sum(duplicated(key)))))

  # rank and top_scores should agree: top_scores[rank] = score_true when rank <= BYOD_N_TOP
  tol   <- 1e-6 * pmax(1, abs(score))
  agree <- ifelse(rank <= BYOD_N_TOP,
                  abs(top[cbind(seq_along(rank), pmin(rank, BYOD_N_TOP))] - score) <= tol,
                  score <= top[, BYOD_N_TOP] + tol)
  if (mean(!agree) > 0.01)
    res$warnings <- sprintf("for %.0f%% of the triples score_true is not at position rank of top_scores: rank and top_scores should be computed in the same setting (raw or filtered)", 100 * mean(!agree))

  res$folds <- fmt_folds(fold)
  res$data  <- tibble(model = info$model, fold = as.integer(fold), head = ent$head, relation = ent$relation,
                      tail = ent$tail, rank = rank, score_true = score, log_z = log_z)
  res$top   <- top
  res
}

# Uploaded files -> data.frame(name, datapath); .zip archives are extracted (.parquet / .csv kept)
byod_expand_uploads <- function(up) {
  out <- lapply(seq_len(nrow(up)), function(i) {
    if (tolower(tools::file_ext(up$name[i])) != "zip")
      return(data.frame(name = up$name[i], datapath = up$datapath[i]))
    dir <- tempfile("byod_zip_")
    utils::unzip(up$datapath[i], exdir = dir)
    fs  <- list.files(dir, recursive = TRUE, full.names = TRUE)
    rel <- substring(fs, nchar(dir) + 1)
    fs  <- fs[!grepl("(^|/)(__MACOSX|\\.)", rel) & tolower(tools::file_ext(fs)) %in% c("parquet", "csv")]
    data.frame(name = basename(fs), datapath = fs)
  })
  do.call(rbind, out)
}

# Training triples (head, relation, tail separated by tabs, no header) -> distinct entities
byod_read_train <- function(path) {
  tr <- utils::read.delim(path, header = FALSE, quote = "", comment.char = "", colClasses = "character",
                          na.strings = character(0), strip.white = TRUE, fileEncoding = "UTF-8-BOM")
  if (ncol(tr) != 3) stop("expected 3 columns separated by tabs (head, relation, tail), found ", ncol(tr))
  unique(c(tr[[1]], tr[[3]]))
}

# Checks across all the uploaded tables and, if everything is fine, computes the metrics.
# Returns list(ok, msgs, results); msgs is a list of (status = "ok" / "warn" / "error", text).
byod_check_all <- function(files, n_entities, train = NULL) {
  msgs <- list()
  add  <- function(status, text) msgs[[length(msgs) + 1]] <<- list(status = status, text = text)
  done <- function() list(ok = !any(vapply(msgs, function(m) m$status == "error", logical(1))),
                          msgs = msgs, results = NULL)

  if (length(files) == 0) { add("error", "no .parquet or .csv file found in the upload"); return(done()) }
  good <- Filter(function(f) length(f$errors) == 0, files)
  if (length(good) < length(files))
    add("error", sprintf("%d of %d file(s) cannot be used: see the table below", length(files) - length(good), length(files)))
  if (length(good) == 0) return(done())
  data <- bind_rows(lapply(good, `[[`, "data"))
  top  <- do.call(rbind, lapply(good, `[[`, "top"))

  # models and folds: each (model, fold) in one file only
  pairs <- bind_rows(lapply(good, function(f) distinct(f$data, model, fold)))
  dup   <- pairs %>% count(model, fold) %>% filter(n > 1)
  if (nrow(dup))
    add("error", paste("the same (model, fold) is in more than one file:", paste0(dup$model, " fold ", dup$fold, collapse = "; ")))
  folds <- lapply(split(pairs$fold, pairs$model), function(f) sort(unique(f)))
  add("ok", paste0(length(folds), " model(s): ", paste0(names(folds), " (", lengths(folds), " folds)", collapse = ", ")))
  if (length(unique(folds)) > 1) add("warn", "the models do not have the same folds")
  if (any(lengths(folds) < 2)) add("warn", "with a single fold the Cross-Fold Analysis and the per-triple intervals are not informative")

  # the same test triples in every (model, fold)
  key      <- paste(data$head, data$relation, data$tail, sep = "\t")
  n_all    <- length(unique(key))
  n_common <- length(unique(Reduce(intersect, split(key, paste(data$model, data$fold)))))
  share    <- n_common / n_all
  if (share == 1) {
    add("ok", sprintf("the same %s test triples in every (model, fold)", fmt_int(n_all)))
  } else if (share >= BYOD_MIN_OVERLAP) {
    add("warn", sprintf("%s of the %s test triples (%.1f%%) are in every (model, fold): the others are kept, but their cross-fold statistics use fewer folds",
                        fmt_int(n_common), fmt_int(n_all), 100 * share))
  } else {
    add("error", sprintf("only %s of the %s test triples (%.0f%%) are in every (model, fold): every fold of every model must be tested on the same test triples",
                         fmt_int(n_common), fmt_int(n_all), 100 * share))
  }
  
  # number of entities
  test_entities <- unique(c(data$head, data$tail))
  if (!is.null(train)) {
    if (!is.null(train$error)) {
      add("warn", paste("training file not used:", train$error))
    } else {
      unseen <- mean(!(test_entities %in% train$entities))
      if (unseen > 0.05)
        add("warn", sprintf("%.0f%% of the entities in the test triples are not in the training file: is it the right file (same entity names)?", 100 * unseen))
    }
  }
  min_n <- max(length(test_entities), max(data$rank), BYOD_N_TOP + 1)
  if (is.null(n_entities) || is.na(n_entities)) {
    add("error", "enter the number of entities |E| (or upload the training file)")
  } else if (n_entities != round(n_entities) || n_entities < min_n) {
    add("error", sprintf("|E| = %s is not valid: it must be an integer, at least the largest rank and the number of entities in the test triples (%s)",
                         format(n_entities), fmt_int(min_n)))
  } else {
    add("ok", sprintf("|E| = %s entities", fmt_int(n_entities)))
  } 
  
  out <- done()
  if (!out$ok) return(out)
    
  # metrics
  per_triple <- bind_cols(
    data %>% transmute(entity_position = rank, fold, mrr = 1 / rank, head, relation, tail, model),
    compute_topk_scores(data$rank, data$score_true, data$log_z, top, n_entities)
  )
  results_matrix <- bind_rows(lapply(split(per_triple, per_triple$model), function(d)
    summarise_by_fold(d) %>% mutate(model = d$model[1])))
  out$results <- list(results_matrix = results_matrix, best_position = per_triple)
  out
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
      .card { border-radius: 10px; box-shadow: 0px 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px; }
      .control-label { font-weight: 600; }
      /* uiOutput() as a block: with display: contents (Shiny >= 1.8.1) recent Shiny versions
         may treat it as hidden and stop updating it after a tab change */
      .shiny-html-output { display: block; }
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
      selectInput("dataset", "Dataset", choices = c(BUILTIN_DATASETS, USER_DATASET)),
      pickerInput(
        inputId = "model", 
        label = "Select Models to Compare", 
        choices = BUILTIN_MODELS,
        selected = "MurE", 
        options = list(`actions-box` = TRUE), 
        multiple = TRUE
      ),
      selectInput("k", "Top-k value", choices = c("1", "3", "10", "20"), selected = "10"),
      sliderInput("max_rank_filter", "Rank Filter (Cross-Fold):", min = 50, max = 1000, value = 500, step = 50),
      
      hr(),
      h5("Inspector Settings"),
      selectInput("model_single", "Model for Detailed Analysis", choices = BUILTIN_MODELS, selected = "MurE"),
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
                     card_header("Hits@k, MRR & Brier Score"), 
                     card_body(shinycssloaders::withSpinner(plotOutput("unified_metrics_plot", height = 350)))
                   )),
                   column(6, card(
                     card_header(textOutput("logk_title")), 
                     card_body(shinycssloaders::withSpinner(plotOutput("logk_forest", height = 350)))
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
                         column(6, plotOutput("tuple_ci_plot_logk", height = 350)),
                         column(6, plotOutput("tuple_ci_plot_mrr", height = 350))
                       )
                     )
                   ))
                 )
        ),

        # ---------- Tab 4: Bring Your Own Data ----------
        tabPanel("Bring Your Own Data",
                 br(),
                 fluidRow(
                   column(5, card(
                     card_header("1. The setting"),
                     card_body(
                       p("Evaluate your own models with the metrics of this demo. The setting is the one of our experiments:"),
                       tags$ul(
                         tags$li(tags$b("one dataset", .noWS = "after"), ": a knowledge graph with a set of entities ", tags$i("E"),
                                 " and a fixed set of ", tags$b("test triples"), " (h, r, t);"),
                         tags$li(tags$b("one or more models", .noWS = "after"), ", e.g. RotatE and ComplEx;"),
                         tags$li("for each model, ", tags$b("several training runs", .noWS = "after"), ", the ", tags$b("folds"),
                                 " (15 in our experiments).")
                       ),
                       p(tags$b("Every fold of every model must be tested on the same test triples."),
                         " This is what allows us to follow each triple across the folds (Cross-Fold Analysis,",
                         " Triple Inspector) and to compare the models on the same ground."),
                       p("For each test triple the model answers the query (h, r, ?) by scoring all the candidate",
                         " tails in ", tags$i("E", .noWS = "after"), ": the softmax of these scores is its predictive distribution.",
                         " You provide the rank of the true tail and a few summaries of the scores; Hits@k, MRR",
                         " and the Top-k Log and Brier scores are computed here.")
                     )
                   )),
                   column(7, card(
                     card_header("2. File format"),
                     card_body(
                       p(tags$b("One table per (model, fold)", .noWS = "after"), ", in Parquet or CSV, named ", tags$code("<model>_fold<f>", .noWS = "after"),
                         ": e.g. ", tags$code("RotatE_fold1.csv", .noWS = "after"), ", \u2026, ", tags$code("RotatE_fold15.csv", .noWS = "after"), ", ",
                         tags$code("ComplEx_fold1.csv", .noWS = "after"), ", \u2026 The model names shown in the app are taken from the file names.",
                         " One row per test triple, with these columns:"),
                       byod_spec_table(),
                       tags$ul(
                         class = "small text-muted",
                         tags$li("Columns are matched by name, in any order; other columns are ignored."),
                         tags$li("Higher score = more plausible (if your model outputs distances, use score = \u2212distance). ",
                                 tags$code("rank"), " and ", tags$code("top_scores"),
                                 " must be computed in the same setting (raw or filtered)."),
                         tags$li("Parquet: ", tags$code("top_scores"), " is a list column. CSV: one string per row, values separated by ';', e.g. ",
                                 tags$code("-17.7;-18.2;-18.9;\u2026")),
                         tags$li("Alternative layout: one table per model, ", tags$code("<model>.parquet"), " or ",
                                 tags$code("<model>.csv", .noWS = "after"), ", with an extra integer column ", tags$code("fold", .noWS = "after"), ".")
                       )
                     )
                   ))
                 ),
                 fluidRow(
                   column(4, card(
                     card_header("3. Upload"),
                     card_body(
                       fileInput("byod_files", "Model outputs: select all the files at once, or one .zip",
                                 multiple = TRUE, accept = c(".parquet", ".csv", ".zip")),
                       fileInput("byod_train", "Training triples (optional)", accept = c(".txt", ".tsv", ".csv")),
                       numericInput("byod_n_entities", "Number of entities |E|", value = NA, min = 1, step = 1),
                       helpText(paste("Filled in from the training file (head, relation, tail separated by tabs, no header)",
                                      "as the number of distinct entities in train \u222A test, or typed by you.",
                                      "The Top-k scores spread the probability mass outside the top-k list",
                                      "over the other |E| \u2212 k entities.")),
                       if (file.exists(BYOD_EXAMPLE)) tagList(
                         hr(),
                         downloadButton("byod_example", "Download example files"),
                         helpText("YAGO3-10, 2 models \u00D7 5 folds, CSV. For this example use |E| = 123,182.")
                       )
                     )
                   )),
                   column(8,
                          card(
                            card_header("4. Checks"),
                            card_body(uiOutput("byod_status"), DTOutput("byod_files_table", fill = FALSE))
                          ),
                          card(
                            card_header("Your results at a glance"),
                            card_body(DTOutput("byod_summary", fill = FALSE))
                          )
                   )
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
  
  # ===== Bring Your Own Data ==================================================

  # 1. read and check every uploaded table (once per upload)
  byod_files <- reactive({
    req(input$byod_files)
    files <- tryCatch(byod_expand_uploads(input$byod_files), error = function(e) e)
    if (inherits(files, "error"))
      return(list(byod_file_error(paste(input$byod_files$name, collapse = ", "),
                                  paste("cannot extract the files:", conditionMessage(files)))))
    lapply(seq_len(NROW(files)), function(i)
      tryCatch(byod_read_file(files$datapath[i], files$name[i]),
               error = function(e) byod_file_error(files$name[i], paste("unexpected error:", conditionMessage(e)))))
  })

  # 2. optional training file -> entities -> |E| = distinct entities in train U test
  byod_train <- reactive({
    req(input$byod_train)
    tryCatch(list(entities = byod_read_train(input$byod_train$datapath), error = NULL),
             error = function(e) list(entities = NULL, error = conditionMessage(e)))
  })
  observe({
    tr <- byod_train()
    req(tr$entities)
    fl <- if (is.null(input$byod_files)) list() else byod_files()
    test_entities <- unlist(lapply(fl, function(f) c(f$data$head, f$data$tail)))
    updateNumericInput(session, "byod_n_entities", value = length(union(tr$entities, test_entities)))
  })
  byod_n <- debounce(reactive(input$byod_n_entities), 700)

  # 3. checks across the tables + metrics
  byod_checked <- reactive({
    req(input$byod_files)
    byod_check_all(byod_files(), byod_n(), train = if (is.null(input$byod_train)) NULL else byod_train())
  })

  # data used by the other tabs when "Your data" is selected (NULL until all the checks pass)
  byod_data <- reactive({
    if (is.null(input$byod_files)) return(NULL)
    ck <- byod_checked()
    if (ck$ok) ck$results else NULL
  })
  byod_models <- reactiveVal(character(0))   # changes only when the model names change
  observe({
    d <- byod_data()
    byod_models(if (is.null(d)) character(0) else unique(d$best_position$model))
  })

  # after a new upload, select "Your data" as soon as the data pass the checks
  byod_pending_switch <- reactiveVal(FALSE)
  observeEvent(input$byod_files, byod_pending_switch(TRUE))
  observe({
    req(byod_pending_switch(), !is.null(byod_data()))
    byod_pending_switch(FALSE)
    updateSelectInput(session, "dataset", selected = USER_DATASET)
    showNotification(paste0("Your data are loaded: '", USER_DATASET, "' is now selected in the Dataset menu."),
                     type = "message", duration = 6)
  })

  output$byod_status <- renderUI({
    if (is.null(input$byod_files)) return(p(class = "text-muted", "Upload your files to run the checks."))
    ck <- byod_checked()
    tagList(
      tags$ul(class = "list-unstyled mb-2", lapply(ck$msgs, function(m) {
        st <- BYOD_STATUS[[m$status]]
        tags$li(style = paste0("color:", st[2], ";"), paste(st[1], m$text))
      })),
      if (ck$ok) {
        div(class = "alert alert-success py-2", "Ready: your models are now available as ",
            tags$b(USER_DATASET, .noWS = "after"), " in the Dataset menu on the left.")
      } else {
        div(class = "alert alert-danger py-2", "Fix the errors and upload the files again.")
      }
    )
  })

  output$byod_files_table <- renderDT({
    req(input$byod_files)
    fl <- byod_files()
    req(length(fl) > 0)
    tab <- data.frame(
      File    = vapply(fl, `[[`, "", "file"),
      Model   = vapply(fl, `[[`, "", "model"),
      Folds   = vapply(fl, `[[`, "", "folds"),
      Triples = vapply(fl, function(f) if (is.null(f$data)) NA_integer_ else nrow(f$data), 1L),
      Status  = vapply(fl, function(f) if (length(f$errors)) "error" else if (length(f$warnings)) "warning" else "ok", ""),
      Notes   = vapply(fl, function(f) paste(c(f$errors, f$warnings, f$notes), collapse = "; "), "")
    )
    datatable(tab, rownames = FALSE,
              options = list(pageLength = 5, lengthMenu = c(5, 15, 60), searching = FALSE, scrollX = TRUE)) %>%
      formatStyle("Status", fontWeight = "bold",
                  color = styleEqual(c("ok", "warning", "error"), c("#1E8449", "#B9770E", "#C0392B")))
  })

  output$byod_summary <- renderDT({
    d <- byod_data()
    validate(need(!is.null(d), "Available when all the checks are passed."))
    k <- as.numeric(input$k)
    tab <- d$best_position %>%
      group_by(Model = model) %>%
      summarise(
        Folds   = n_distinct(fold),
        Triples = n_distinct(head, relation, tail),
        MRR     = mean(mrr),
        Hits    = mean(entity_position <= k),
        LogS    = mean(.data[[paste0("k_", k)]]),
        Brier   = mean(.data[[paste0("orig_brier_", k)]]),
        .groups = "drop"
      ) %>%
      mutate(across(c(MRR, Hits, LogS, Brier), ~round(.x, 3))) %>%
      rename(!!paste0("Hits@", k) := Hits, !!paste0("Top-", k, " Log") := LogS, !!paste0("Top-", k, " Brier") := Brier)
    datatable(tab, rownames = FALSE, options = list(dom = "t", ordering = FALSE),
              caption = "Means over all the test triples and folds; k is set in the sidebar.")
  })

  output$byod_example <- downloadHandler(
    filename = function() BYOD_EXAMPLE,
    content  = function(file) file.copy(BYOD_EXAMPLE, file),
    contentType = "application/zip"
  )

  # ===== Data of the selected dataset =========================================

  # models offered in the sidebar: the built-in ones, or the ones read from the uploaded files
  observeEvent(list(input$dataset, byod_models()), {
    user   <- identical(input$dataset, USER_DATASET)
    models <- if (user) byod_models() else BUILTIN_MODELS
    sel    <- intersect(input$model, models)
    if (length(sel) == 0) sel <- if (user) models else "MurE"
    sel_single <- if (isTRUE(input$model_single %in% models)) input$model_single else if (user) head(models, 1) else "MurE"
    freezeReactiveValue(input, "model")
    freezeReactiveValue(input, "model_single")
    updatePickerInput(session, "model", choices = models, selected = sel)
    updateSelectInput(session, "model_single", choices = models, selected = sel_single)
  }, ignoreInit = TRUE, priority = 10)

  # results of one model: list(results_matrix, best_position), NULL if not available
  model_data <- function(dataset, m) {
    if (identical(dataset, USER_DATASET)) {
      d <- byod_data()
      if (is.null(d) || !(m %in% d$best_position$model)) return(NULL)
      return(list(results_matrix = filter(d$results_matrix, model == m),
                  best_position  = filter(d$best_position, model == m)))
    }
    res <- tryCatch(load_builtin(dataset, m), error = function(e) NULL)
    if (is.null(res)) return(NULL)
    res$results_matrix$model <- m
    res$best_position$model  <- m
    res
  }
  need_user_data <- function() {
    if (identical(input$dataset, USER_DATASET))
      validate(need(!is.null(byod_data()),
                    if (is.null(input$byod_files)) "No data loaded yet: upload your files in the 'Bring Your Own Data' tab."
                    else "Your files did not pass the checks: see the 'Bring Your Own Data' tab."))
  }

  # Reactive Data Loading (models to compare)
  processed_data <- reactive({
    req(input$dataset)
    need_user_data()
    req(input$model)
    all_models_list <- compact(map(input$model, ~ model_data(input$dataset, .x)))
    validate(need(length(all_models_list) > 0, paste0("No results available for ", input$dataset, ".")))
    list(
      results_matrix = bind_rows(map(all_models_list, "results_matrix")),
      best_position = bind_rows(map(all_models_list, "best_position"))
    )
  })
  
  res_matrix <- reactive({ processed_data()$results_matrix })
  
  # model for the detailed analysis (independent of the models to compare)
  single_data <- reactive({
    req(input$dataset)
    need_user_data()
    req(input$model_single)
    d <- model_data(input$dataset, input$model_single)
    validate(need(!is.null(d), paste0("No results available for ", input$model_single, " on ", input$dataset, ".")))
    d
  })
  res_matrix_single <- reactive({ single_data()$results_matrix })
  best_pos_single   <- reactive({ single_data()$best_position })
  
  bp_sampled_reactive <- reactive({
    req(best_pos_single(), sampled_triples())
    best_pos_single() %>%
      semi_join(sampled_triples(), by = c("head", "relation", "tail"))
  })

  # folds of the reference-fold menu: the ones in the data
  observe({
    folds <- as.character(sort(unique(best_pos_single()$fold)))
    req(length(folds) > 0)
    sel <- shiny::isolate(input$ref_fold)   # shiny:: because library(renv) masks isolate()
    updateSelectInput(session, "ref_fold", choices = folds, selected = if (isTRUE(sel %in% folds)) sel else folds[1])
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
    y_max_data <- max(plot_data$ci_high, plot_data$mean_val, na.rm = TRUE)
    
    y_upper <- max(1, ceiling(y_max_data * 10) / 10)  # arrotonda a 0.1 sopra
    
    ggplot(plot_data, aes(x = factor(fold), y = mean_val, color = model, shape = metric_label)) +
      geom_point(position = position_dodge(width = 0.8), size = 3.5) +
      geom_errorbar(aes(ymin = ci_low, ymax = ci_high), 
                    position = position_dodge(width = 0.8), width = 0.4, alpha = 0.6) +
      scale_y_continuous(
        limits = c(0, y_upper),
        breaks = seq(0, y_upper, 0.1)
      ) +
      labs(
        title = "Accuracy & Calibration Comparison",
        subtitle = paste0("Metrics evaluated at k = ", k_val),
        x = "Fold", 
        y = "Value", 
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
  # new sample when "Fetch Triples" is pressed or the data of the inspected model change
  sampled_triples <- eventReactive(list(input$resample, best_pos_single()), {
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
    } else if (input$triple_filter == "High Variance") {
      valid_triples <- triple_stats %>% arrange(desc(sd_rank)) %>% head(150)
    } else if (input$triple_filter == "Medium (Rank near 50)") {
      # Safest approach: grab top 150 closest to rank 50, then sample from them
      valid_triples <- triple_stats %>% arrange(abs(mean_rank - 50)) %>% head(150)
    } else {  # "Random (Mixed)"
      valid_triples <- triple_stats
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
    
    datatable(
  stat, 
  options = list(
    pageLength = 5,           # Default rows shown
    lengthMenu = c(5, 10, 15), # Dropdown options
    searching = FALSE,        # Disables the search bar
    scrollX = TRUE            # Keeps your horizontal scroll
  ),
  rownames = FALSE
)
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
      select(fold, entity_position, k_1, k_3, k_10, k_20) %>%
      mutate(across(starts_with("k_"), ~round(.x, 4)))
    
    datatable(triple_rows, options = list(pageLength = 5, scrollX = TRUE,searching = FALSE, lengthMenu = list(c(5, 10), c("5", "10"))), 
              colnames = c("Fold", "Rank","Log-1", "Log-3", "Log-10", "Log-20"),
              rownames = FALSE)
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
      coord_cartesian(ylim = c(0, 14)) + 
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
  
  output$download_combined_csv <- downloadHandler(
    filename = function() paste0("metrics_", gsub("[^A-Za-z0-9]+", "_", input$dataset), "_", input$model_single, "_", Sys.Date(), ".csv"),
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
        subtitle = paste0("Reference Fold = ", ref_fold),
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
