args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  pattern <- paste0("--", name, "=")
  hit <- args[grepl(pattern, args)]
  if (length(hit) == 0) return(default)
  sub(pattern, "", hit[1])
}

source("R/forecast_module.R")

data_path <- get_arg("data", "data.xlsx")
artifacts_dir <- get_arg("artifacts", "artifacts")
output_path <- get_arg("output", "artifacts/imae_forecast_12m.csv")
horizon <- as.integer(get_arg("horizon", "12"))

forecast_df <- generate_imae_projection(
  data_path = data_path,
  artifacts_dir = artifacts_dir,
  horizon = horizon,
  output_path = output_path
)

print(forecast_df)
