args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  pattern <- paste0("--", name, "=")
  hit <- args[grepl(pattern, args)]
  if (length(hit) == 0) return(default)
  sub(pattern, "", hit[1])
}

source("R/modeling_module.R")

data_path <- get_arg("data", "data.xlsx")
out_dir <- get_arg("out", "artifacts")
assess <- as.integer(get_arg("assess", "12"))

result <- train_and_evaluate_models(data_path = data_path, output_dir = out_dir, assess = assess)
print(result$evaluation)
cat(sprintf("Modelo seleccionado: %s\n", result$metadata$selected_model))
