suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(lubridate)
  library(glmnet)
  library(randomForest)
  library(jsonlite)
})

find_target_column <- function(df) {
  target_candidates <- names(df)[grepl("imae", names(df), ignore.case = TRUE)]
  if (length(target_candidates) == 0) {
    stop("No se encontró una columna objetivo que contenga 'IMAE'.")
  }
  target_candidates[1]
}

find_date_column <- function(df) {
  date_candidates <- names(df)[grepl("fecha|date", names(df), ignore.case = TRUE)]
  if (length(date_candidates) == 0) return(NA_character_)
  date_candidates[1]
}

load_imae_data <- function(path = "data.xlsx") {
  if (!file.exists(path)) stop(sprintf("No se encontró el archivo: %s", path))

  raw <- read_excel(path)
  if (nrow(raw) == 0) stop("El archivo Excel está vacío.")

  names(raw) <- trimws(names(raw))
  date_col <- find_date_column(raw)

  if (!is.na(date_col)) {
    raw[[date_col]] <- as.Date(raw[[date_col]])
    raw <- raw %>% arrange(.data[[date_col]])
    rownames(raw) <- as.character(raw[[date_col]])
    raw[[date_col]] <- NULL
  } else {
    idx <- seq.Date(from = as.Date("2000-01-01"), by = "month", length.out = nrow(raw))
    rownames(raw) <- as.character(idx)
  }

  target <- find_target_column(raw)

  numeric_df <- raw %>% mutate(across(everything(), ~ suppressWarnings(as.numeric(.x))))
  numeric_df <- numeric_df %>% filter(if_any(everything(), ~ !is.na(.x)))

  if (all(is.na(numeric_df[[target]]))) {
    stop("La columna objetivo IMAE no contiene datos numéricos válidos.")
  }

  list(data = numeric_df, target = target)
}

make_rolling_splits <- function(n, initial = NULL, assess = 12, skip = 0) {
  if (is.null(initial)) initial <- max(24, floor(n * 0.6))
  splits <- list()
  i <- 1
  end_train <- initial

  while ((end_train + assess) <= n) {
    train_idx <- seq_len(end_train)
    test_idx <- seq(from = end_train + 1, to = end_train + assess)
    splits[[i]] <- list(train = train_idx, test = test_idx)
    i <- i + 1
    end_train <- end_train + assess + skip
  }

  if (length(splits) == 0) stop("No se pudieron crear particiones de validación temporal.")
  splits
}

compute_metrics <- function(actual, pred) {
  mae <- mean(abs(actual - pred), na.rm = TRUE)
  rmse <- sqrt(mean((actual - pred)^2, na.rm = TRUE))
  sst <- sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)
  sse <- sum((actual - pred)^2, na.rm = TRUE)
  r2 <- ifelse(sst == 0, NA_real_, 1 - (sse / sst))
  c(mae = mae, rmse = rmse, r2 = r2)
}

fit_predict_model <- function(model_name, x_train, y_train, x_test) {
  if (model_name == "linear_regression") {
    train_df <- data.frame(y = y_train, x_train)
    fit <- lm(y ~ ., data = train_df)
    pred <- as.numeric(predict(fit, newdata = as.data.frame(x_test)))
    return(list(model = fit, pred = pred))
  }

  if (model_name == "ridge") {
    fit <- cv.glmnet(as.matrix(x_train), y_train, alpha = 0, family = "gaussian")
    pred <- as.numeric(predict(fit, newx = as.matrix(x_test), s = "lambda.min"))
    return(list(model = fit, pred = pred))
  }

  if (model_name == "random_forest") {
    fit <- randomForest(x = as.data.frame(x_train), y = y_train, ntree = 500, mtry = max(1, floor(sqrt(ncol(x_train)))))
    pred <- as.numeric(predict(fit, newdata = as.data.frame(x_test)))
    return(list(model = fit, pred = pred))
  }

  stop(sprintf("Modelo no soportado: %s", model_name))
}

extract_importance <- function(model_name, fitted_model, feature_names) {
  if (model_name == "random_forest") {
    imp <- randomForest::importance(fitted_model, type = 1)
    values <- imp[, 1]
    names(values) <- rownames(imp)
    return(sort(values, decreasing = TRUE))
  }

  if (model_name == "ridge") {
    coefs <- as.matrix(coef(fitted_model, s = "lambda.min"))
    coefs <- abs(coefs[-1, 1])
    names(coefs) <- rownames(as.matrix(coef(fitted_model, s = "lambda.min")))[-1]
    return(sort(coefs, decreasing = TRUE))
  }

  if (model_name == "linear_regression") {
    coefs <- abs(stats::coef(fitted_model)[-1])
    coefs <- coefs[feature_names]
    coefs[is.na(coefs)] <- 0
    return(sort(coefs, decreasing = TRUE))
  }

  numeric()
}

train_and_evaluate_models <- function(data_path = "data.xlsx", output_dir = "artifacts", assess = 12) {
  bundle <- load_imae_data(data_path)
  df <- bundle$data
  target <- bundle$target

  complete_df <- df %>% filter(if_all(everything(), ~ !is.na(.x)))
  x <- complete_df %>% select(-all_of(target))
  y <- complete_df[[target]]

  if (ncol(x) == 0) stop("No hay indicadores explicativos disponibles tras limpiar los datos.")

  splits <- make_rolling_splits(n = nrow(complete_df), assess = assess)
  model_names <- c("linear_regression", "ridge", "random_forest")
  results <- list()

  for (model_name in model_names) {
    fold_metrics <- lapply(splits, function(sp) {
      x_train <- x[sp$train, , drop = FALSE]
      y_train <- y[sp$train]
      x_test <- x[sp$test, , drop = FALSE]
      y_test <- y[sp$test]

      fit_obj <- fit_predict_model(model_name, x_train, y_train, x_test)
      compute_metrics(y_test, fit_obj$pred)
    })

    metric_matrix <- do.call(rbind, fold_metrics)
    avg <- colMeans(metric_matrix, na.rm = TRUE)

    results[[model_name]] <- data.frame(
      model = model_name,
      mae = unname(avg["mae"]),
      rmse = unname(avg["rmse"]),
      r2 = unname(avg["r2"])
    )
  }

  evaluation <- bind_rows(results) %>% arrange(rmse)
  best_model_name <- evaluation$model[1]

  final_fit <- fit_predict_model(best_model_name, x, y, x)
  importance <- extract_importance(best_model_name, final_fit$model, names(x))

  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  saveRDS(final_fit$model, file.path(output_dir, "best_imae_model.rds"))
  write.csv(evaluation, file.path(output_dir, "model_evaluation.csv"), row.names = FALSE)

  importance_df <- data.frame(indicador = names(importance), importancia = as.numeric(importance))
  write.csv(importance_df, file.path(output_dir, "feature_importance.csv"), row.names = FALSE)

  metadata <- list(
    target = target,
    selected_model = best_model_name,
    n_obs = nrow(complete_df),
    features = names(x),
    top_indicadores = head(importance_df, 10)
  )

  write_json(metadata, file.path(output_dir, "model_metadata.json"), pretty = TRUE, auto_unbox = TRUE)

  invisible(list(evaluation = evaluation, metadata = metadata))
}
