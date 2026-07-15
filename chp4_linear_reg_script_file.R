# =============================================================================
# OULAD: Linear Regression with tidymodels
# Predicting avg_score (a student's average assessment score)
# =============================================================================


# =============================================================================
# DATA DICTIONARY: OULAD
# =============================================================================
#
# IDENTIFIERS
#   id_student          Unique student identifier
#
# COURSE CONTEXT
#   code_module         Course module code (AAA-GGG, 7 modules)
#   code_presentation   Presentation period (e.g., "2013J" = 2013 Fall)
#   domain              Subject area - STEM or Social Science
#                       (perfectly determined by code_module)
#
# STUDENT DEMOGRAPHICS
#   gender              M or F
#   region              UK region of residence
#   highest_education   Prior education level (No Formal quals, Lower Than A
#                       Level, A Level, HE Qualification, Post Graduate)
#   imd_band            Index of Multiple Deprivation band, 10 bands
#                       (0-10% most deprived, 90-100% least deprived)
#   age_band            Age at registration: 0-35, 35-55, or 55<=
#   disability          Whether student declared a disability (Y/N)
#
# ENROLLMENT
#   num_of_prev_attempts        Number of prior attempts at this module
#   studied_credits             Credits enrolled for this presentation
#   date_registration           Day of registration relative to module start
#                               (negative = registered before start)
#   date_unregistration         Day of unregistration if withdrew (NA if
#                               completed) - LEAKAGE
#   module_presentation_length  Length of module in days
#
# ENGAGEMENT (aggregated over the presentation)
#   total_clicks        Total clicks on the virtual learning environment
#   total_days_active   Distinct days with any activity
#   n_assessments       Number of assessments the student took
#
# ASSESSMENT PERFORMANCE
#   avg_score           Average assessment score (0-100) - #REGRESSION TARGET
#   min_score           Lowest single assessment score - #LEAKAGE
#   max_score           Highest single assessment score - #LEAKAGE
#   n_failed            Count of assessments failed - #LEAKAGE
#   avg_days_early      Average days submitted early (negative = late)
#
# OUTCOME #NA at time of prediction 
#   final_result        Pass / Distinction / Fail / Withdrawn
#                       (used to derive wf)
#   wf                  Binary classification target:
#                       "Successful" (Pass/Distinction) vs "WF" (Fail/Withdrawn)
#
# =============================================================================


library(tidyverse)
library(tidymodels)
library(skimr)
library(caret)
library(corrr) # correlation check
library(rcompanion) # correlation check
library(DALEXtra) #explain models

set.seed(1234)


# -----------------------------------------------------------------------------
# 1. Load and prepare the data
# -----------------------------------------------------------------------------
# Several columns come in as character because of missing values in the source
# file; we coerce them to numeric here so the recipe can work with them correctly.

oulad_raw <- read_csv("oulad_clean.csv")

oulad <- oulad_raw %>%
  mutate(
    avg_score         = as.numeric(avg_score),
    avg_days_early    = as.numeric(avg_days_early),
    n_assessments     = as.numeric(n_assessments),
    total_clicks      = as.numeric(total_clicks),
    total_days_active = as.numeric(total_days_active),
    date_registration = as.numeric(date_registration)
  )

# Peek at what we have
oulad %>% slice_head(n = 5)
oulad %>% summarise(n_missing_target = sum(is.na(avg_score)))
glimpse(oulad)
summary(oulad)
skim(oulad)






#___________________
#examine the target

# How much of the target is missing?
# Anything predicting avg_score can only be trained on rows where
# avg_score is observed.
oulad %>%
  summarise(
    n_rows            = n(),
    n_missing_target  = sum(is.na(avg_score)),
    pct_missing_target = mean(is.na(avg_score)) * 100
  )


# Sanity-check the outcome's range and distribution
# avg_score should be roughly 0–100. Values outside that would signal a
# data problem worth investigating before modeling.
oulad %>%
  summarise(
    min_score   = min(avg_score, na.rm = TRUE),
    max_score   = max(avg_score, na.rm = TRUE),
    mean_score  = mean(avg_score, na.rm = TRUE),
    median_score = median(avg_score, na.rm = TRUE)
  )


#visually inspect
outcome_distribution_plot <- oulad %>%
  filter(!is.na(avg_score)) %>%
  ggplot(aes(x = avg_score)) +
  geom_histogram(bins = 40) +
  labs(
    title = "Distribution of avg_score",
    subtitle = "Bounded roughly 0–100; check for weirdness at the edges",
    x = "avg_score",
    y = "Count"
  ) +
  theme_minimal()

# mean less than median means negative (left) skew
# The bulk of students score relatively high
# A long tail stretches to the left (low scores)
# A minority of students score much worse than typical, 
    # and they pull the mean down below the median
outcome_distribution_plot





#____________________________
# check nzv and zv
#Filtering decisions are often made before the split
# TRUE returns the full diagnostic table instead of just column indices
nzv_report <- oulad %>%
  select(-avg_score) %>%   # exclude the outcome; we only care about predictors
  nearZeroVar(saveMetrics = TRUE) %>%
  as_tibble(rownames = "variable")

nzv_report %>% print(n = Inf)









# -----------------------------------------------------------------------------
# 2. Drop rows with a missing target
# -----------------------------------------------------------------------------
# We can only train on rows where the outcome is observed. Imputing the target
# would be inventing labels, which is not the same problem as imputing a
# predictor. Do this outside the recipe.

oulad_model_data <- oulad %>%
  filter(!is.na(avg_score))

glimpse(oulad_model_data)






# -----------------------------------------------------------------------------
# 3. Drop leakage and non-predictor columns
# -----------------------------------------------------------------------------
#
#   final_result, wf     : maybe not helpful at time of prediction.
#   min_score, max_score : mathematical siblings of the mean → target leakage
#   n_failed             : an assessment is "failed" based on its score → leaks
#   id_student           : an identifier, not a feature
#   date_unregistration  : recorded only after the outcome is known → leakage

oulad_predictors <- oulad_model_data %>%
  select(
    -final_result, -wf,
    -min_score, -max_score, -n_failed,
    -id_student, -date_unregistration
  )

# Sanity check
oulad_predictors %>% slice_head(n = 5)
glimpse(oulad_predictors)





# Check for redundant predictors before the split
#____________________________________________
# ---- Numeric pairwise correlation tests -------------------------------------
# Uses cor.test() to get p-values and confidence intervals for each pair.

oulad_predictors %>%
  select(where(is.numeric), -avg_score) %>%
  cor(use = "pairwise.complete.obs") %>%
  round(2)


#check for linear relationships:
# Pearson's r is a number between -1 and +1 that measures the strength 
# and direction of the linear relationship between two variables.
cor(oulad %>% select(where(is.numeric)), 
    use = "pairwise.complete.obs")[, "avg_score"]

cor(oulad)

# -----------------------------------------------------------------------------
# 4. Train / test split
# -----------------------------------------------------------------------------
# We stratify on the target so both splits have a similar score distribution.
# rsample automatically bins a numeric strata variable into quartiles.

oulad_split <- initial_split(
  oulad_predictors,
  prop  = 0.8,
  strata = avg_score
)

train_data <- training(oulad_split)
test_data  <- testing(oulad_split)



#space exists here too for working the data (what will the model train on)
# nzv and zv (again)
# Filtering decisions can be made after the split based on what the model will actually
  # see.
# saveMetrics = TRUE returns the full diagnostic table rather than just
# the column indices that would be dropped
nzv_report <- train_data %>%
  select(-avg_score) %>%                    # exclude the outcome
  nearZeroVar(saveMetrics = TRUE) %>%
  as_tibble(rownames = "variable") %>%
  arrange(desc(freqRatio))                  # most suspicious first

# See everything (default tibble printing truncates)
nzv_report %>% print(n = Inf)






# -----------------------------------------------------------------------------
# 5. Cross-validation folds (built from training data only)
# -----------------------------------------------------------------------------
# Even with no hyperparameters to tune, CV gives us a more stable estimate of
# out-of-sample performance than a single train/test split.

cv_folds <- vfold_cv(train_data, v = 5, strata = avg_score)







# -----------------------------------------------------------------------------
# 6. Recipe (preprocessing blueprint)
# -----------------------------------------------------------------------------
# We follow the ordering from our preprocessing rec from Boehmke

reg_recipe <- recipe(avg_score ~ ., data = train_data) %>%
  
  # Keep domain in the data but exclude it from modeling
  # (perfectly determined by code_module → causes issues with model)
  update_role(domain, new_role = "id") %>%
  
  # 6a. Drop near-zero-variance numeric predictors before spending effort on them
  step_nzv(all_numeric_predictors()) %>%
  
  # 6b. Impute missing values (median for numeric, mode for categorical)
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  
  # 6c. Fix skew in numeric predictors (Yeo-Johnson handles zeros and negatives)
        # YJ strives for a bell curve.
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # 6d. Lump rare categorical levels before dummy encoding to avoid a flood of
  #     near-empty dummy columns
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  
  # 6e. Center and scale numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  # 6f. Encode categoricals last so we don't destroy sparsity earlier
  step_dummy(all_nominal_predictors())

# Peek at what the recipe produces on a small sample
reg_recipe %>%
  prep() %>%
  bake(new_data = NULL) %>%
  slice_head(n = 5)


# -----------------------------------------------------------------------------
# 7. Model specification
# -----------------------------------------------------------------------------
# Ordinary least squares linear regression — no hyperparameters to tune.
# Note the mode is now "regression" not"classification".

lm_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")


# -----------------------------------------------------------------------------
# 8. Workflow (recipe + model spec bundled together)
# -----------------------------------------------------------------------------

lm_workflow <- workflow() %>%
  add_recipe(reg_recipe) %>%
  add_model(lm_spec)


# -----------------------------------------------------------------------------
# 9. Evaluate with cross-validation
# -----------------------------------------------------------------------------
# For regression we use RMSE (error in target units), MAE (median-friendly error
# in target units), and R-squared (proportion of variance explained).

reg_metrics <- metric_set(rmse, mae, rsq)

cv_results <- lm_workflow %>%
  fit_resamples(
    resamples = cv_folds,
    metrics   = reg_metrics
  )

cv_summary <- collect_metrics(cv_results)
cv_summary


# Output
# A tibble: 3 × 6
# .metric .estimator   mean     n std_err .config        
# <chr>     <chr>      <dbl>   <int>   <dbl> <chr>          
# 1 mae     standard   10.6       5  0.0531  pre0_mod0_post0
# 2 rmse    standard   14.2       5  0.0747  pre0_mod0_post0
# 3 rsq     standard   0.255      5  0.00418 pre0_mod0_post0


# MAE - mean absolute error - On average the model is off by about 10.6 points
# RMSE - root mean square error - On average the model is off by about 14.2 points
# R squared - The model explains about 25.5% of the variance in avg_score.
  # ~74% is unexplained by the predictors.








# -----------------------------------------------------------------------------
# 10. Fit the final model on all training data
# -----------------------------------------------------------------------------
#4/5ths of the data were used for each 5 fold resample, now we use all of it.
final_fit <- lm_workflow %>%
  fit(data = train_data)









# -----------------------------------------------------------------------------
# 11. Evaluate on the held-out test set
# -----------------------------------------------------------------------------

test_predictions <- final_fit %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data %>% select(avg_score))

# look at 5 students in the test data and see how the model did:
test_predictions %>% slice_head(n = 5)

#    .pred avg_score
#    <dbl>     <dbl>
#  1  74.7      68  # Over estimated by 6.7%


test_performance <- test_predictions %>%
  reg_metrics(truth = avg_score, estimate = .pred)

test_performance








# -----------------------------------------------------------------------------
# 12. Variable importance and partial dependence plots 
# -----------------------------------------------------------------------------
# VIP and PDP are both called "global explainers"
#> Global explanations — describe the model's behavior across the entire dataset. 
#> They answer "how does this model work in general?"



# the VIP package is in a strange state now so we use workarounds.




#_____________________
#tidyverse extraction
#(tstat = coefficient estimate / SE of that estimate)
# "how big is the effect, relative to how uncertain we are about it?"
# 0 = Uncncertain of the contribution
final_fit %>%
  extract_fit_parsnip() %>%              # pull the fitted lm object out of the workflow
  tidy() %>% # convert coefficients to a tibble (term, estimate, std.error, statistic, p.value)
  filter(term != "(Intercept)") %>%      # drop the intercept — not a predictor
  slice_max(abs(statistic), n = 10) %>%  # keep the 10 predictors with the largest |t-statistic|
  ggplot(aes(abs(statistic),             # x-axis: importance
             fct_reorder(term,           # y-axis: predictor names, reordered so biggest bars sit on top
                         abs(statistic)))) +
  geom_col()                             # horizontal bars



#Use dalextra explainers

explainer <- explain_tidymodels(
  final_fit,
  data  = train_data %>% select(-avg_score),
  y     = train_data$avg_score
)


# "How much worse would the model's predictions get if this predictor were useless?"
#uses RMSE not Tstat so the results are different
vip_result <- model_parts(explainer)
plot(vip_result, max_vars = 10)



#partial dependence plot (PDP)
# A PDP shows how the model's predicted outcome changes as we vary one
# predictor across its range, while averaging over the values of all other
# predictors. It answers: "holding everything else roughly equal, how does
# this predictor drive the prediction?"

pdp_result <- model_profile(explainer, variables = "total_clicks")

plot(pdp_result)




#single student predict:
#What variables to account for?
glimpse(train_data)

# Make up a single student to predict for
new_student <- tibble(
  code_module               = "BBB",
  code_presentation         = "2013J",
  gender                    = "F",
  region                    = "Scotland",
  highest_education         = "HE Qualification",
  imd_band                  = "40-50%",
  age_band                  = "35-55",
  num_of_prev_attempts      = 0,
  studied_credits           = 60,
  disability                = "N",
  date_registration         = -30,
  n_assessments             = 5,
  avg_days_early            = 3,
  module_presentation_length = 240,
  total_clicks              = 1200,
  total_days_active         = 60,
  domain                    = "Social Science"
)

# Predict
predict(final_fit, new_data = new_student)


#What?
#You can see the coefficients here: 
final_fit


#What is a coefficient?
#> A coefficient is the weight the model multiplies a predictor by, telling you 
#> how much the prediction changes for each one-unit increase in that predictor, 
#> holding all other predictors constant.

#What do these calculations look like?
# Numbers are plugged in and math happens. 
# A student with 0 previous attempts and 60 credits:
                
    #intercept  #prev attmepts   #studied credits   
# y^ = 77.51 +( −0.164 × 0) + (−0.172 × 60) =

# 77.51 + 0 − 10.32 =

# 67.19 #avg score prediction






# -----------------------------------------------------------------------------
# Export objects for downstream use (Shiny app, reports, reproducibility)
# -----------------------------------------------------------------------------

# The fitted workflow: recipe + fitted model bundled together. The Shiny app
# calls predict(final_fit, new_data = user_input) on raw user inputs and gets
# a prediction back — no preprocessing needed on the app side.
saveRDS(final_fit, "final_fit.rds")





# The full modeling dataset (pre-split, post-cleanup): all the raw-shaped rows
# with just the model's predictors and outcome. Useful for the Shiny app to
# populate valid ranges for sliders and valid levels for dropdowns, and to
# show summary stats or context alongside predictions.
saveRDS(oulad_predictors, "oulad_predictors.rds")

