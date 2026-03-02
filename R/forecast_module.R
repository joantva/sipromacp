suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(jsonlite)
})

source("R/modeling_module.R")

forecast_feature <- function(series, horizon = 12, dates = NULL) {
  s <- series[!is.na(series)]
  if (length(s) == 0) return(rep(NA_real_, horizon))
  if (length(s) < 24) return(rep(tail(s, 1), horizon))

  if (is.null(dates)) {
    dates <- seq.Date(from = as.Date("2000-01-01"), by = "month", length.out = length(series))
  }

  valid_dates <- dates[!is.na(series)]
  month_vals <- month(valid_dates)
  monthly_means <- tapply(s, month_vals, mean, na.rm = TRUE)

  if (length(monthly_means) < 12) {
    return(rep(tail(s, 1), horizon))
  }

  trend <- lm(s ~ seq_along(s))
  slope <- coef(trend)[2]

  last_date <- max(valid_dates)
  out <- numeric(horizon)
  for (h in seq_len(horizon)) {
    next_month <- month(last_date %m+% months(h))
    seasonal <- monthly_means[as.character(next_month)]
    out[h] <- seasonal + slope * h
  }
  out
}

build_future_features <- function(df, feature_cols, horizon = 12) {
  history_dates <- as.Date(rownames(df))
  if (any(is.na(history_dates))) {
    history_dates <- seq.Date(from = as.Date("2000-01-01"), by = "month", length.out = nrow(df))
  }

  start_date <- floor_date(max(history_dates), unit = "month") %m+% months(1)
  future_dates <- seq.Date(from = start_date, by = "month", length.out = horizon)

  future <- data.frame(fecha = future_dates)
  for (col in feature_cols) {
    future[[col]] <- forecast_feature(df[[col]], horizon = horizon, dates = history_dates)
  }

  future
}

predict_with_model <- function(model_name, model_obj, future_x) {
  x_mat <- as.matrix(future_x)

  if (model_name == "ridge") {
    return(as.numeric(predict(model_obj, newx = x_mat, s = "lambda.min")))
  }

  if (model_name == "linear_regression") {
    return(as.numeric(predict(model_obj, newdata = as.data.frame(future_x))))
  }

  if (model_name == "random_forest") {
    return(as.numeric(predict(model_obj, newdata = as.data.frame(future_x))))
  }

  stop(sprintf("Modelo no soportado para proyección: %s", model_name))
}

generate_imae_projection <- function(data_path = "data.xlsx", artifacts_dir = "artifacts", horizon = 12,
                                     output_path = "artifacts/imae_forecast_12m.csv") {
  bundle <- load_imae_data(data_path)
  metadata <- fromJSON(file.path(artifacts_dir, "model_metadata.json"))
  model_obj <- readRDS(file.path(artifacts_dir, "best_imae_model.rds"))

  features <- metadata$features
  future <- build_future_features(bundle$data, features, horizon = horizon)

  future_x <- future %>% select(all_of(features))
  preds <- predict_with_model(metadata$selected_model, model_obj, future_x)

  out <- future %>% mutate(imae_proyectado = preds)
  write.csv(out, output_path, row.names = FALSE)
  out
}
