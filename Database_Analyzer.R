# =============================================================================
# Shiny App for Machine Learning Model Analysis
# =============================================================================

# Load required packages
required_packages <- c("shiny", "shinydashboard", "tidyverse", "randomForest", 
                       "caret", "vip", "iml", "pdp", "gridExtra", "doParallel", 
                       "pROC", "DALEX", "DT", "shinyjs", "shinyWidgets")

# Install missing packages
missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if(length(missing_packages)) install.packages(missing_packages)

# Load all packages
lapply(required_packages, library, character.only = TRUE)

# Enable shinyjs
useShinyjs()

# Source original functions (assuming they are in a separate file)
# source("functions.R") # Uncomment and adjust path if needed
preprocess_data <- function(df, target_var, predictor_vars) {
  cat("Starting data preprocessing...\n")
  print(summary(df))
  
  # Identify numeric and categorical columns
  num_cols <- predictor_vars[sapply(df[predictor_vars], is.numeric)]
  char_cols <- predictor_vars[sapply(df[predictor_vars], function(x) is.character(x) | is.factor(x))]
  
  # Fill numeric columns with median
  if(length(num_cols) > 0) {
    df[num_cols] <- df[num_cols] %>% 
      mutate(across(all_of(num_cols), ~ replace_na(., median(., na.rm = TRUE))))
  }
  
  # Fill categorical columns with mode
  if(length(char_cols) > 0) {
    df[char_cols] <- df[char_cols] %>% 
      mutate(across(all_of(char_cols), ~ {
        lev <- na.omit(.)
        replace_na(., names(sort(table(.), decreasing = TRUE))[1])
      }))
  }
  
  # Ensure target variable is a factor
  df[[target_var]] <- factor(df[[target_var]], ordered = TRUE)
  
  # Remove rows where target variable is NA
  df <- df[!is.na(df[[target_var]]), ]
  
  cat("Data preprocessing completed. Data dimensions:", dim(df), "\n")
  return(df)
}

# Model training and cross-validation function
train_rf_model <- function(df, target_var, predictor_vars, mtry_grid, cv_folds, seed) {
  cat("Starting model training and cross-validation...\n")
  
  # Set up cross-validation
  ctrl <- trainControl(
    method = "cv",
    number = cv_folds,
    classProbs = TRUE,
    summaryFunction = defaultSummary,
    savePredictions = "final"
  )
  
  # Define tuning grid
  rf_grid <- expand.grid(mtry = mtry_grid)
  
  # Create formula
  formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Train model
  set.seed(seed)
  rf_cv <- train(
    model_formula,
    data = df,
    method = "rf",
    trControl = ctrl,
    tuneGrid = rf_grid,
    importance = TRUE
  )
  
  cat("Model training completed. Best mtry:", rf_cv$bestTune$mtry, "\n")
  return(rf_cv)
}

# Modified model evaluation function - using cross-validation results
evaluate_model <- function(rf_model, df, target_var) {
  cat("Starting model evaluation (based on cross-validation)...\n")
  
  # Get cross-validation predictions
  cv_predictions <- rf_model$pred
  
  # Ensure predictions are in the same order as original data
  cv_predictions <- cv_predictions[order(cv_predictions$rowIndex),]
  
  # Extract predictions for best mtry
  best_mtry <- rf_model$bestTune$mtry
  best_predictions <- cv_predictions[cv_predictions$mtry == best_mtry,]
  
  # Create confusion matrix
  actual_values <- best_predictions$obs
  predicted_values <- best_predictions$pred
  conf_matrix <- confusionMatrix(predicted_values, actual_values)
  
  # Extract prediction probabilities
  prob_columns <- grep("^[^.]+\\.", names(best_predictions), value = TRUE)
  pred_probs <- best_predictions[, prob_columns]
  
  # Rename columns to match level names if necessary
  if(!all(gsub("^[^.]+\\.", "", prob_columns) == levels(actual_values))) {
    names(pred_probs) <- levels(actual_values)
  }
  
  # Extract performance metrics
  accuracy <- conf_matrix$overall["Accuracy"]
  kappa <- conf_matrix$overall["Kappa"]
  
  # Print results
  cat("=== Cross-Validation Model Performance Evaluation ===\n")
  cat("Best mtry parameter:", best_mtry, "\n")
  cat("Cross-validation accuracy:", round(accuracy, 4), "\n")
  cat("Cross-validation Kappa:", round(kappa, 4), "\n")
  
  print(conf_matrix)
  
  return(list(
    confusion_matrix = conf_matrix,
    cv_predictions = best_predictions,
    probabilities = pred_probs,
    accuracy = accuracy,
    kappa = kappa
  ))
}

# Fixed ROC curve plotting function
plot_roc_curves <- function(rf_model, target_var) {
  cat("Plotting ROC curves (based on cross-validation)...\n")
  
  # Get cross-validation predictions
  cv_predictions <- rf_model$pred
  
  # Extract predictions for best mtry
  best_mtry <- rf_model$bestTune$mtry
  best_predictions <- cv_predictions[cv_predictions$mtry == best_mtry,]
  
  # Extract actual values
  actual_values <- best_predictions$obs
  class_levels <- levels(actual_values)
  
  # Debug: Print column names to help identify probability columns
  cat("Available column names:", paste(names(best_predictions), collapse=", "), "\n")
  
  # More flexible identification of probability columns - try multiple formats
  prob_columns <- character(0)
  
  # Try method 1: Standard caret format - e.g., "High", "Medium", "Low"
  if(length(prob_columns) == 0 && all(class_levels %in% names(best_predictions))) {
    prob_columns <- class_levels
    cat("Identified probability columns using method 1:", paste(prob_columns, collapse=", "), "\n")
  }
  
  # Try method 2: Dotted prefix format - e.g., "class.High", "class.Medium"
  if(length(prob_columns) == 0) {
    pattern_columns <- grep("\\.", names(best_predictions), value = TRUE)
    if(length(pattern_columns) > 0) {
      # Extract part after dot and check if it matches classes
      suffixes <- unique(sapply(strsplit(pattern_columns, "\\."), function(x) x[length(x)]))
      if(all(class_levels %in% suffixes)) {
        prob_columns <- pattern_columns
        cat("Identified probability columns using method 2:", paste(prob_columns, collapse=", "), "\n")
      }
    }
  }
  
  # Try method 3: Look for columns containing "prob"
  if(length(prob_columns) == 0) {
    prob_pattern_cols <- grep("prob|Prob|PROB", names(best_predictions), value = TRUE)
    if(length(prob_pattern_cols) == length(class_levels)) {
      prob_columns <- prob_pattern_cols
      cat("Identified probability columns using method 3:", paste(prob_columns, collapse=", "), "\n")
    }
  }
  
  # Try last resort: Assume column positions
  if(length(prob_columns) == 0) {
    # Check how many columns and make best guess
    total_cols <- ncol(best_predictions)
    standard_cols <- c("rowIndex", "mtry", "Resample", "obs", "pred") # Standard columns
    remaining_cols <- setdiff(names(best_predictions), standard_cols)
    
    if(length(remaining_cols) >= length(class_levels)) {
      prob_columns <- remaining_cols[1:length(class_levels)]
      cat("Identified probability columns using last resort:", paste(prob_columns, collapse=", "), "\n")
    }
  }
  
  # Confirm if probability columns were found
  if(length(prob_columns) == 0) {
    stop("Unable to identify prediction probability columns, please check model output format")
  }
  
  if(length(prob_columns) != length(class_levels)) {
    warning("Number of identified probability columns does not match number of classes, results may be inaccurate")
  }
  
  # Extract prediction probabilities
  pred_probs <- as.data.frame(best_predictions[, prob_columns, drop=FALSE])
  
  # Plot ROC curves
  roc_list <- list()
  auc_values <- numeric(length(class_levels))
  
  par(mfrow = c(1, 1))
  for (i in 1:length(class_levels)) {
    binary_outcome <- ifelse(actual_values == class_levels[i], 1, 0)
    
    # Ensure i does not exceed number of pred_probs columns
    if(i <= ncol(pred_probs)) {
      roc_obj <- roc(binary_outcome, pred_probs[, i], quiet = TRUE)
      
      if (i == 1) {
        plot(roc_obj, col = i, main = paste("Cross-Validation ROC Curves -", target_var), 
             lwd = 2, legacy.axes = TRUE)
      } else {
        plot(roc_obj, col = i, add = TRUE, lwd = 2)
      }
      
      roc_list[[i]] <- roc_obj
      auc_values[i] <- auc(roc_obj)
    } else {
      warning(paste("Class", class_levels[i], "has no corresponding probability column"))
      auc_values[i] <- NA
    }
  }
  
  legend("bottomright", 
         legend = paste0(class_levels, " (AUC = ", round(auc_values, 2), ")"), 
         col = 1:length(class_levels), lwd = 2)
  
  # Calculate macro-average AUC
  macro_auc <- mean(auc_values, na.rm = TRUE)
  cat("\n AUC (all classes):", auc_values, "\n AUC (average across classes):", round(macro_auc, 4), "\n")
  
  return(list(roc_objects = roc_list, auc_values = auc_values, macro_auc = macro_auc))
}

# Variable importance analysis function
analyze_variable_importance <- function(rf_model, df, target_var, predictor_vars, seed) {
  cat("Analyzing variable importance...\n")
  
  # Train final model
  formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  set.seed(seed)
  final_rf <- randomForest(
    model_formula,
    data = df,
    mtry = rf_model$bestTune$mtry,
    importance = TRUE
  )
  
  # Extract importance
  importance_data <- importance(final_rf)
  importance_df <- data.frame(
    Variable = rownames(importance_data),
    MeanDecreaseAccuracy = importance_data[, 1],
    MeanDecreaseGini = importance_data[, 2]
  ) %>%
    arrange(desc(MeanDecreaseAccuracy))
  
  # Output importance ranking
  print("=== Variable Importance Ranking ===")
  print(importance_df)
  
  # Plot importance using ggplot2
  importance_long <- rbind(
    data.frame(
      Variable = importance_df$Variable,
      Value = importance_df$MeanDecreaseAccuracy,
      Metric = "Mean Decrease Accuracy"
    ),
    data.frame(
      Variable = importance_df$Variable,
      Value = importance_df$MeanDecreaseGini,
      Metric = "Mean Decrease Gini"
    )
  )
  
  importance_plot <- ggplot(importance_long, aes(x = reorder(Variable, Value), y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    theme_classic() +
    scale_fill_manual(values = c("Mean Decrease Accuracy" = "grey50", "Mean Decrease Gini" = "grey80")) +
    labs(
      title = "Variable Importance",
      subtitle = "Random Forest Feature Importance Metrics",
      x = "Predictor Variable",
      y = "Importance"
    ) +
    theme(
      text = element_text(size = 12, family = "Times"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  print(importance_plot)
  
  return(list(final_model = final_rf, importance_df = importance_df, importance_plot = importance_plot))
}

# Fixed PDP+ICE combined plot function - plot separately
create_pdp_ice_combined <- function(final_rf, df, importance_df, predictor_vars, target_var, top_n = 8) {
  cat("Creating PDP+ICE combined plots...\n")
  
  top_predictors <- head(importance_df$Variable, top_n)
  
  # Create predictor - include all predictor variables
  predictor <- Predictor$new(
    model = final_rf, 
    data = df[, predictor_vars, drop = FALSE], # Use all predictor variables
    y = df[[target_var]], # Explicitly specify target variable
    type = "prob"
  )
  
  combined_plots <- list()
  
  for (var in top_predictors) {
    cat("Processing variable:", var, "\n")
    
    tryCatch({
      # Check variable type
      is_categorical <- is.factor(df[[var]]) || is.character(df[[var]])
      grid_size <- if(is_categorical) {
        min(length(unique(df[[var]])), 8)
      } else {
        15
      }
      
      # Create PDP+ICE combined effect
      combined_effect <- FeatureEffect$new(
        predictor = predictor,
        feature = var,
        method = "pdp+ice", # Display both PDP and ICE
        grid.size = grid_size
      )
      
      # Create plot
      combined_plot <- combined_effect$plot() +
        labs(title = paste("PDP + ICE:", var),
             subtitle = "PDP (average effect), ICE (individual effect)",
             y = "Prediction probability") +
        theme_minimal() +
        theme(plot.title = element_text(size = 12, face = "bold"),
              plot.subtitle = element_text(size = 10),
              axis.text = element_text(size = 10),
              axis.title = element_text(size = 11))
      
      combined_plots[[var]] <- combined_plot
      
      # Show single plot
      cat("  Showing", var, "'s PDP+ICE combined figure\n")
      print(combined_plot)
      cat("\n") 
      
      # Save plot to a PDF file with a unique filename
      timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
      pdf_filename <- paste0("pdp_ice_", var, "_", timestamp, ".pdf")
      pdf(file = pdf_filename, width = 8, height = 6)
      print(combined_plot)
      dev.off()
      cat("  Saved", var, "'s PDP+ICE plot to", pdf_filename, "\n")
      
    }, error = function(e) {
      cat("  Variable", var, "'s PDP+ICE figure error:", e$message, "\n")
    })
  }
  
  
  cat("All PDP+ICE combined figures completed\n")
  
  return(combined_plots)
}








# learning curve
analyze_learning_curve <- function(df, target_var, predictor_vars, best_mtry, seed) {
  cat("Analyzing learning curve...\n")
  
  formula_str <- paste(target_var, "~", paste(predictor_vars, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  learning_curve_data <- data.frame()
  
  for(trees in seq(50, 1000, by = 50)) {
    set.seed(seed)
    temp_rf <- randomForest(
      model_formula,
      data = df,
      mtry = best_mtry,
      ntree = trees
    )
    
    learning_curve_data <- rbind(learning_curve_data, 
                                 data.frame(ntree = trees, 
                                            OOB_Error = tail(temp_rf$err.rate[,1], 1)))
  }
  
  
  p <- ggplot(learning_curve_data, aes(x = ntree, y = OOB_Error)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red", size = 2) +
    labs(title = paste("Learning Curve (mtry =", best_mtry, ")"),
         x = "Number of Trees",
         y = "Out-of-Bag Error") +
    theme_minimal()
  
  print(p)
  
  return(learning_curve_data)
}


plot_roc_curves <- function(rf_model, target_var) {
  cat("Plotting ROC Curves (Cross-Validation)...\n")
  
  # Get cross-validation predictions
  cv_predictions <- rf_model$pred
  if (nrow(cv_predictions) == 0) {
    stop("No predictions available in rf_model$pred")
  }
  
  # Extract best mtry predictions
  best_mtry <- rf_model$bestTune$mtry
  best_predictions <- cv_predictions[cv_predictions$mtry == best_mtry, ]
  if (nrow(best_predictions) == 0) {
    stop("No predictions available for best mtry:", best_mtry)
  }
  
  # Ensure predictions are ordered
  best_predictions <- best_predictions[order(best_predictions$rowIndex), ]
  
  # Extract actual values and class levels
  actual_values <- best_predictions$obs
  class_levels <- levels(actual_values)
  if (length(class_levels) < 2) {
    stop("Target variable must have at least two levels for ROC analysis")
  }
  
  # Identify probability columns
  prob_columns <- grep("^prob\\.", names(best_predictions), value = TRUE)
  if (length(prob_columns) == 0) {
    prob_columns <- class_levels[class_levels %in% names(best_predictions)]
  }
  
  # Validate probability columns
  if (length(prob_columns) == 0) {
    stop("No probability columns found in predictions")
  }
  
  # Match probability columns to class levels
  prob_labels <- gsub("^prob\\.", "", prob_columns)
  if (!all(class_levels %in% prob_labels)) {
    missing_classes <- class_levels[!class_levels %in% prob_labels]
    warning("Missing probability columns for classes: ", paste(missing_classes, collapse = ", "))
    class_levels <- class_levels[class_levels %in% prob_labels]
    prob_columns <- prob_columns[prob_labels %in% class_levels]
  }
  
  if (length(class_levels) == 0) {
    stop("No valid probability columns match the class levels")
  }
  
  # Extract prediction probabilities
  pred_probs <- as.data.frame(best_predictions[, prob_columns, drop = FALSE])
  colnames(pred_probs) <- class_levels  # Fixed typo here
  
  # Initialize ROC and AUC storage
  roc_list <- list()
  auc_values <- numeric(length(class_levels))
  plot_data <- list()
  
  # Calculate ROC for each class
  for (i in seq_along(class_levels)) {
    tryCatch({
      binary_outcome <- ifelse(actual_values == class_levels[i], 1, 0)
      if (length(unique(binary_outcome)) < 2) {
        warning("Class ", class_levels[i], " has only one level in binary outcome, skipping ROC")
        auc_values[i] <- NA
        next
      }
      if (nrow(pred_probs) != length(binary_outcome)) {
        stop("Mismatch in rows: pred_probs (", nrow(pred_probs), ") vs binary_outcome (", length(binary_outcome), ")")
      }
      roc_obj <- roc(binary_outcome, pred_probs[, i], quiet = TRUE)
      roc_list[[class_levels[i]]] <- roc_obj
      auc_values[i] <- auc(roc_obj)
      
      # Store data for plotting
      plot_data[[i]] <- data.frame(
        FPR = 1 - roc_obj$specificities,
        TPR = roc_obj$sensitivities,
        Class = class_levels[i],
        AUC = round(auc_values[i], 2)
      )
    }, error = function(e) {
      warning("ROC calculation failed for class ", class_levels[i], ": ", e$message)
      auc_values[i] <- NA
    })
  }
  
  # Combine plot data
  plot_data <- bind_rows(plot_data)
  
  # Create publication-quality ROC plot
  p <- ggplot(plot_data, aes(x = FPR, y = TPR, color = Class, linetype = Class)) +
    geom_line(size = 1) +
    theme_classic() +
    scale_color_grey() +
    labs(title = paste("ROC Curves for", target_var),
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
    annotate("text", x = 0.75, y = 0.25, 
             label = paste("AUC:", paste(unique(plot_data$Class), 
                                         "(", unique(plot_data$AUC), ")", collapse = "\n")),
             size = 4, hjust = 0) +
    theme(
      text = element_text(size = 12, family = "Times"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      legend.title = element_blank()
    )
  
  print(p)
  
  # Calculate macro-average AUC
  macro_auc <- mean(auc_values, na.rm = TRUE)
  cat("\nAUC (Per Class):", auc_values, "\nMacro-Average AUC:", round(macro_auc, 4), "\n")
  
  return(list(roc_objects = roc_list, auc_values = auc_values, macro_auc = macro_auc, roc_plot = p))
}




# Function to compute SHAP values using DALEX
compute_shap_values <- function(final_model, df, predictor_vars, target_var) {
  cat("Starting SHAP values calculation...\n")
  
  # Create explainer for classification
  explainer <- DALEX::explain(
    model = final_model,
    data = df[, predictor_vars],
    y = df[[target_var]],
    predict_function = function(m, d) predict(m, d, type = "prob"),
    label = "Random Forest",
    type = "classification"
  )
  
  # Compute SHAP for each instance
  shap_list <- lapply(1:nrow(df), function(i) {
    shap <- DALEX::predict_parts(
      explainer = explainer,
      new_observation = df[i, predictor_vars],
      type = "shap",
      B = 10  # Reduced for speed; increase for more accuracy
    )
    shap$instance <- i
    shap
  })
  
  shap_df <- do.call(rbind, shap_list)
  
  # Parse variable into variable_name
  shap_df <- shap_df %>%
    mutate(variable_name = sub(" =.*", "", variable))
  
  # Add feature_value from original data (as character)
  feature_values <- do.call(rbind, lapply(1:nrow(df), function(i) {
    data.frame(
      instance = i,
      variable_name = predictor_vars,
      feature_value = as.character(unlist(df[i, predictor_vars]))
    )
  }))
  
  shap_df <- left_join(shap_df, feature_values, by = c("instance", "variable_name"))
  
  cat("SHAP values calculation completed.\n")
  
  # Save shap_df to a CSV file locally
  write.csv(shap_df, file = "shap_values.csv", row.names = FALSE)
  cat("SHAP values saved to shap_values.csv\n")
  
  return(shap_df)
}

# Function to create SHAP dependence plots
create_shap_dependence_plots <- function(shap_df, df, predictor_vars, target_var) {
  cat("Creating SHAP dependence plots...\n")
  
  # Filter out baseline or special rows
  shap_df <- shap_df %>% filter(!grepl("^_", variable_name))
  
  vars <- unique(shap_df$variable_name)
  
  dep_plots <- list()
  
  for (v in vars) {
    cat("Processing variable:", v, "\n")
    
    data_v <- shap_df %>% filter(variable_name == v)
    
    is_num <- is.numeric(df[[v]])
    
    if (is_num) {
      data_v <- data_v %>% mutate(feature_num = as.numeric(feature_value))
      
      p <- ggplot(data_v, aes(x = feature_num, y = contribution)) +
        geom_point(alpha = 0.5) +
        geom_smooth(span=0.4, method = "loess", se = TRUE) +
        facet_wrap(~label) +
        theme_classic() +
        labs(title = paste("SHAP Dependence for", v),
             subtitle = "With LOESS fitting curve",
             x = v, y = "SHAP Value") +
        theme(
          text = element_text(size = 12, family = "Times"),
          plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 12, hjust = 0.5),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10)
        )
    } else {
      p <- ggplot(data_v, aes(x = feature_value, y = contribution)) +
        geom_boxplot() +
        facet_wrap(~label) +
        theme_classic() +
        labs(title = paste("SHAP Dependence for", v),
             subtitle = "Boxplot showing median, 25% and 75% quantiles",
             x = v, y = "SHAP Value") +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1),
          text = element_text(size = 12, family = "Times"),
          plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
          plot.subtitle = element_text(size = 12, hjust = 0.5),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10)
        )
    }
    
    print(p)
    dep_plots[[v]] <- p
  }
  
  cat("SHAP dependence plots completed.\n")
  
  return(dep_plots)
}

# Modified run_complete_analysis
run_complete_analysis <- function(df, target_var = "Liver_MP_accum",
                                  predictor_vars = c(
                                    "MP_component_polymer", "MP_size_um", "MP_shape",
                                    "mouse_rat", "exposure_method", "exposure_duration_day",
                                    "MP_dose_mg_kg_dw_day", "if_coexposure", 
                                    "MP_accumulation_detection_method"
                                  ),
                                  seed = 2025,
                                  cv_folds = 10,
                                  mtry_grid = c(2, 3, 4, 5, 6, 7)) {
  
  cat("=============================================================================\n")
  cat("Starting Machine Learning Analysis Pipeline\n")
  cat("Target Variable:", target_var, "\n")
  cat("Number of Predictors:", length(predictor_vars), "\n")
  cat("=============================================================================\n")
  
  # 1. Preprocess Data
  df_clean <- preprocess_data(df, target_var, predictor_vars)
  
  # 2. Model Training and Cross-Validation
  rf_model <- train_rf_model(df_clean, target_var, predictor_vars, mtry_grid, cv_folds, seed)
  print(rf_model)
  
  # 3. Model Evaluation
  evaluation_results <- evaluate_model(rf_model, df_clean, target_var)
  
  # 4. ROC Curves
  roc_results <- plot_roc_curves(rf_model, target_var)
  
  # 5. Variable Importance Analysis
  importance_results <- analyze_variable_importance(rf_model, df_clean, target_var, predictor_vars, seed)
  
  # 6. Combined PDP + ICE Plots
  combined_pdp_ice <- create_pdp_ice_combined(importance_results$final_model,
                                              df_clean,
                                              importance_results$importance_df,
                                              predictor_vars,
                                              target_var)
  
  # 7. Learning Curve
  learning_curve_data <- analyze_learning_curve(df_clean, target_var, predictor_vars, 
                                                rf_model$bestTune$mtry, seed)
  
  # 8. SHAP Values
  shap_results <- compute_shap_values(importance_results$final_model, df_clean, predictor_vars, target_var)
  
  # 9. SHAP Dependence Plots
  shap_dependence <- create_shap_dependence_plots(shap_results, df_clean, predictor_vars, target_var)
  
  # 10. SHAP Wide Table
  shap_wide <- shap_results %>%
    filter(!grepl("^_", variable_name)) %>%
    mutate(key = paste(variable_name, label, sep = "_")) %>%
    select(instance, key, contribution) %>%
    pivot_wider(names_from = key, values_from = contribution)
  
  df_wide <- df_clean %>%
    mutate(instance = row_number()) %>%
    left_join(shap_wide, by = "instance") %>%
    select(-instance)
  
  # Save SHAP values with original data to local CSV
  shap_csv_path <- "shap_values_with_original_data.csv"
  write_csv(df_wide, shap_csv_path)
  cat("SHAP values with original data saved locally to:", shap_csv_path, "\n")
  
  # 11. Summary Report
  cat("\n=============================================================================\n")
  cat("Analysis Completed! Summary of Results:\n")
  cat("=============================================================================\n")
  cat("Target Variable:", target_var, "\n")
  cat("Best mtry Parameter:", rf_model$bestTune$mtry, "\n")
  cat("Cross-Validation Accuracy:", round(max(rf_model$results$Accuracy), 4), "\n")
  cat("Top 3 Most Important Variables:\n")
  print(head(importance_results$importance_df$Variable, 3))
  cat("=============================================================================\n")
  
  # Return all results
  return(list(
    data = df_clean,
    cv_model = rf_model,
    final_model = importance_results$final_model,
    evaluation = evaluation_results,
    importance = importance_results$importance_df,
    roc_results = roc_results,
    learning_curve = learning_curve_data,
    combined_pdp_ice = combined_pdp_ice,
    shap = shap_results,
    shap_wide = df_wide,
    shap_dependence = shap_dependence
  ))
}

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Machine Learning Analysis Dashboard"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data Input", tabName = "data", icon = icon("upload")),
      menuItem("Model Configuration", tabName = "config", icon = icon("cog")),
      menuItem("Results", tabName = "results", icon = icon("chart-bar"))
    )
  ),
  dashboardBody(
    tags$head(
      tags$link(rel = "stylesheet", href = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css")
    ),
    tabItems(
      tabItem(tabName = "data",
              fluidRow(
                box(width = 12, title = "Data Input", status = "primary", solidHeader = TRUE,
                    fileInput("file", "Upload CSV File", accept = ".csv"),
                    textInput("data_path", "Or Enter Data Path", 
                              value = "D:\\CU\\meta_agent\\MP_bioaccumulation_rat_mice\\model_building_agent_based\\final_extracted_MP_bioaccu.csv"),
                    actionButton("load_data", "Load Data", class = "bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded")
                )
              ),
              fluidRow(
                box(width = 12, title = "Data Preview", status = "info", solidHeader = TRUE,
                    DTOutput("data_preview")
                )
              )
      ),
      tabItem(tabName = "config",
              fluidRow(
                box(width = 6, title = "Variable Selection", status = "primary", solidHeader = TRUE,
                    selectInput("target_var", "Select Target Variable", choices = NULL),
                    selectizeInput("predictor_vars", "Select Predictor Variables", 
                                   choices = NULL, multiple = TRUE),
                    actionButton("update_vars", "Update Variables", class = "bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded")
                ),
                box(width = 6, title = "Model Parameters", status = "primary", solidHeader = TRUE,
                    numericInput("seed", "Random Seed", value = 2025, min = 1),
                    numericInput("cv_folds", "Cross-Validation Folds", value = 10, min = 2, max = 20),
                    numericInput("cv_repeats", "Cross-Validation Repeats", value = 1, min = 1, max = 10),
                    textInput("mtry_grid", "mtry Grid (comma-separated)", value = "2,3,4,5,6,7"),
                    actionButton("run_analysis", "Run Analysis", class = "bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded")
                )
              )
      ),
      tabItem(tabName = "results",
              fluidRow(
                tabBox(width = 12, title = "Analysis Results",
                       tabPanel("Model Summary",
                                verbatimTextOutput("model_summary"),
                                plotOutput("model_plot", height = "400px")
                       ),
                       tabPanel("Evaluation",
                                verbatimTextOutput("evaluation_summary"),
                                plotOutput("confusion_matrix", height = "400px")
                       ),
                       tabPanel("ROC Curves",
                                plotOutput("roc_plot", height = "400px")
                       ),
                       tabPanel("Variable Importance",
                                plotOutput("importance_plot", height = "400px"),
                                DTOutput("importance_table")
                       ),
                       tabPanel("PDP + ICE",
                                uiOutput("pdp_ice_plots")
                       ),
                       tabPanel("Learning Curve",
                                plotOutput("learning_curve", height = "400px")
                       ),
                       tabPanel("SHAP Analysis",
                                plotOutput("shap_summary_plot", height = "400px"),
                                downloadButton("download_shap", "Download SHAP CSV"),
                                uiOutput("shap_dependence_plots")
                       )
                )
              )
      )
    )
  )
)

# Define Server
server <- function(input, output, session) {
  
  # Reactive values to store data and results
  rv <- reactiveValues(
    df = NULL,
    results = NULL
  )
  
  # Load data
  observeEvent(input$load_data, {
    withProgress(message = "Loading data...", value = 0, {
      tryCatch({
        if (!is.null(input$file)) {
          rv$df <- read_csv(input$file$datapath)
        } else if (!is.null(input$data_path) && file.exists(input$data_path)) {
          rv$df <- read_csv(input$data_path)
        } else {
          stop("No valid file uploaded or file path provided.")
        }
        
        # Update variable selectors
        updateSelectInput(session, "target_var", choices = names(rv$df))
        updateSelectizeInput(session, "predictor_vars", 
                             choices = names(rv$df), 
                             selected = c(
                               "MP_component_polymer", "MP_size_um", "MP_shape",
                               "mouse_rat", "exposure_method", "exposure_duration_day",
                               "MP_dose_mg_kg_dw_day", "if_coexposure", 
                               "MP_accumulation_detection_method"
                             ))
        
        # Show data preview
        output$data_preview <- renderDT({
          datatable(head(rv$df, 10), options = list(scrollX = TRUE))
        })
        
        showNotification("Data loaded successfully!", type = "message")
      }, error = function(e) {
        showNotification(paste("Error loading data:", e$message), type = "error")
      })
    })
  })
  
  # Run analysis
  observeEvent(input$run_analysis, {
    req(rv$df, input$target_var, input$predictor_vars)
    
    withProgress(message = "Running analysis...", value = 0, {
      tryCatch({
        # Parse mtry grid
        mtry_grid <- as.numeric(unlist(strsplit(input$mtry_grid, ",")))
        
        # Run the complete analysis with the loaded data frame
        rv$results <- run_complete_analysis(
          df = rv$df,
          target_var = input$target_var,
          predictor_vars = input$predictor_vars,
          seed = input$seed,
          cv_folds = input$cv_folds,
          mtry_grid = mtry_grid
        )
        
        # Render Model Summary Plot
        output$model_plot <- renderPlot({
          ggplot(data = rv$results$cv_model$results, aes(x = mtry, y = Accuracy)) +
            geom_line(size = 1) +
            geom_point(size = 2) +
            theme_classic() +
            labs(title = "Model Performance Across mtry Values",
                 x = "mtry",
                 y = "Cross-Validation Accuracy") +
            theme(
              text = element_text(size = 12, family = "Times"),
              plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 10)
            )
        })
        
        output$model_summary <- renderPrint({
          print(rv$results$cv_model)
        })
        
        # Render Evaluation Summary
        output$evaluation_summary <- renderPrint({
          print(rv$results$evaluation$confusion_matrix)
          cat("\nAccuracy:", round(rv$results$evaluation$accuracy, 4), "\n")
          cat("Kappa:", round(rv$results$evaluation$kappa, 4), "\n")
        })
        
        # Render Confusion Matrix
        output$confusion_matrix <- renderPlot({
          cm <- as.table(rv$results$evaluation$confusion_matrix$table)
          cm_df <- as.data.frame(cm)
          ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
            geom_tile(color = "black") +
            geom_text(aes(label = Freq), color = "black", size = 4) +
            scale_fill_gradient(low = "white", high = "grey50") +
            theme_classic() +
            labs(title = "Confusion Matrix",
                 x = "Predicted Class",
                 y = "Actual Class") +
            theme(
              text = element_text(size = 12, family = "Times"),
              plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 10)
            )
        })
        
        # Render ROC Plot
        output$roc_plot <- renderPlot({
          rv$results$roc_results$roc_plot
        })
        
        # Render Variable Importance Plot
        output$importance_plot <- renderPlot({
          imp_df <- rv$results$importance
          ggplot(imp_df, aes(x = reorder(Variable, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
            geom_bar(stat = "identity", fill = "grey50", color = "black") +
            coord_flip() +
            theme_classic() +
            labs(title = "Variable Importance",
                 x = "Predictor Variable",
                 y = "Mean Decrease in Accuracy") +
            theme(
              text = element_text(size = 12, family = "Times"),
              plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 10)
            )
        })
        
        output$importance_table <- renderDT({
          datatable(rv$results$importance, 
                    options = list(scrollX = TRUE),
                    colnames = c("Variable", "Mean Decrease Accuracy", "Mean Decrease Gini"))
        })
        
        # Render PDP + ICE Plots
        output$pdp_ice_plots <- renderUI({
          combined_plots <- rv$results$combined_pdp_ice
          plot_output_list <- lapply(names(combined_plots), function(var) {
            plotOutput(paste0("pdp_ice_", var), height = "400px")
          })
          do.call(tagList, plot_output_list)
        })
        
        observe({
          combined_plots <- rv$results$combined_pdp_ice
          for (var in names(combined_plots)) {
            local({
              var_name <- var
              output[[paste0("pdp_ice_", var_name)]] <- renderPlot({
                combined_plots[[var_name]] +
                  theme_classic() +
                  theme(
                    text = element_text(size = 12, family = "Times"),
                    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
                    plot.subtitle = element_text(size = 12, hjust = 0.5),
                    axis.title = element_text(size = 12),
                    axis.text = element_text(size = 10)
                  )
              })
            })
          }
        })
        
        # Render Learning Curve
        output$learning_curve <- renderPlot({
          ggplot(rv$results$learning_curve, aes(x = ntree, y = OOB_Error)) +
            geom_line(size = 1, color = "black") +
            geom_point(size = 2, color = "black") +
            theme_classic() +
            labs(title = paste("Learning Curve (mtry =", rv$results$cv_model$bestTune$mtry, ")"),
                 x = "Number of Trees",
                 y = "Out-of-Bag Error") +
            theme(
              text = element_text(size = 12, family = "Times"),
              plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 10)
            )
        })
        
        # Render SHAP Summary Plot
        output$shap_summary_plot <- renderPlot({
          shap_df <- rv$results$shap %>%
            filter(!grepl("^_", variable_name))
          
          shap_summary <- shap_df %>%
            group_by(variable_name) %>%
            summarise(mean_abs_shap = mean(abs(contribution))) %>%
            arrange(desc(mean_abs_shap))
          
          ggplot(shap_summary, aes(x = reorder(variable_name, mean_abs_shap), y = mean_abs_shap)) +
            geom_bar(stat = "identity", fill = "grey50") +
            coord_flip() +
            theme_classic() +
            labs(title = "SHAP Feature Importance",
                 x = "Variable",
                 y = "Mean |SHAP Value|") +
            theme(
              text = element_text(size = 12, family = "Times"),
              plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 10)
            )
        })
        
        
        # Render SHAP Dependence Plots
        output$shap_dependence_plots <- renderUI({
          dep_plots <- rv$results$shap_dependence
          plot_output_list <- lapply(names(dep_plots), function(var) {
            plotOutput(paste0("dep_", var), height = "400px")
          })
          do.call(tagList, plot_output_list)
        })
        
        observe({
          dep_plots <- rv$results$shap_dependence
          for (var in names(dep_plots)) {
            local({
              var_name <- var
              output[[paste0("dep_", var_name)]] <- renderPlot({
                dep_plots[[var_name]]
              })
            })
          }
        })
        
        showNotification("Analysis completed successfully!", type = "message")
      }, error = function(e) {
        showNotification(paste("Error running analysis:", e$message), type = "error")
      })
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)