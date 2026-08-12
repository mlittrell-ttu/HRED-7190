#═════════════════════════════════════════════════════════════════════════════
# Complete App
#  They sit next to each other so each page reads as one self-contained unit.
#  At the bottom, an "assembler" ui lists the pages and an assembler server
#  calls each page's server.
#
#    1. SETUP
#    2. PREDICT PAGE   (ui + server)   -- finished
#    3. EXAMINE PAGE   (ui + server)   finished
#    4. ABOUT PAGE     (ui only)
#    5. ASSEMBLE       ui + server + launch
#
#
#═════════════════════════════════════════════════════════════════════════════


#═════════════════════════════════════════════════════════════════════════════
#  1. SETUP
#═════════════════════════════════════════════════════════════════════════════

library(shiny)
library(bslib)
library(tidyverse)
library(tidymodels)
library(DT)
library(bsicons)
library(xgboost)

## The trained model -- loaded once when the app starts.
final_fit <- readRDS("final_fit_gbm.rds")


## Module -> domain lookup vector to label the course variables.
module_domain <- c(
  AAA = "Social Sciences", BBB = "Social Sciences",
  CCC = "STEM",            DDD = "STEM",
  EEE = "STEM",            FFF = "STEM",
  GGG = "Social Sciences"
)


## The "typical student" -- median (numeric) / most-common (categorical)
## values from the full OULAD. The Examine comparison chart measures the built
## student against this baseline.
typical_student <- tibble(
  code_module = "BBB",
  code_presentation = "2014J",
  gender = "M",
  region = "Scotland",
  highest_education = "A Level or Equivalent",
  imd_band = "20-30%",
  age_band = "0-35",
  disability = "N",
  num_of_prev_attempts = 0,
  studied_credits = 60,
  date_registration = -57,
  n_failed = 0,
  avg_days_early = 0,
  total_clicks = 740,
  total_days_active = 47,
  module_presentation_length = 262
)

#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________




####PREDICT####

#═════════════════════════════════════════════════════════════════════════════
# 2. PREDICT PAGE  (finished -- built in the earlier lessons)
#  Upload a CSV, score it with the model, explore the predictions.
#═════════════════════════════════════════════════════════════════════════════

#_____________________________________________________________________________
## Predict -- UI
#_____________________________________________________________________________
predict_page <- nav_panel(
  "Predict",
  
  ## ___ Upload (top, full width) ___
  card(
    card_header("Load data"),
    fileInput("upload", "Upload a student CSV:", accept = ".csv")
  ),
  
  ## ___ Filters (left) + content (right) ___
  layout_columns(
    col_widths = c(3, 9),
    
    ## ___ Filter panel ___
    card(
      card_header("Filters"),
      selectInput("region", "Region:",
                  choices = c("All",
                              "East Anglian Region", "East Midlands Region",
                              "Ireland", "London Region", "North Region",
                              "North Western Region", "Scotland",
                              "South East Region", "South Region",
                              "South West Region", "Wales",
                              "West Midlands Region", "Yorkshire Region")),
      selectInput("gender", "Gender:",
                  choices = c("All", "M", "F")),
      selectInput("age_band", "Age band:",
                  choices = c("All", "0-35", "35-55", "55<=")),
      sliderInput("threshold", "High-risk threshold:",
                  min = 0, max = 1, value = 0.70, step = 0.05)
    ),
    
    ## ___ Content column ___
    tags$div(
      
      ## Score cards
      layout_columns(
        col_widths = c(4, 4, 4),
        value_box(title = "Students scored",
                  value = textOutput("n_scored"),
                  showcase = bsicons::bs_icon("people"),
                  theme    = value_box_theme(bg = "#dce8e0", fg = "#3a3f42"),
                  p("in view")),
        value_box(title = "Predicted WF",
                  value = textOutput("n_wf"),
                  showcase = bsicons::bs_icon("exclamation-triangle"),
                  theme    = value_box_theme(bg = "#f0ccc2", fg = "#3a3f42"),
                  p("of students in view")),
        value_box(title = "High risk",
                  value = textOutput("n_high_risk"),
                  showcase = bsicons::bs_icon("exclamation-octagon"),
                  theme    = value_box_theme(bg = "#e8b3a6", fg = "#3a3f42"),
                  p("above the threshold"))
      ),
      
      ## Charts
      layout_columns(
        col_widths = c(6, 6),
        card(card_header("Risk distribution"),
             plotOutput("risk_hist")),
        card(card_header("WF rate by module"),
             plotOutput("wf_by_module"))
      ),
      
      ## Table with download
      card(
        card_header("Scored students",
                    downloadButton("download_scored", "Download CSV",
                                   class = "btn-sm")),
        DTOutput("scored_table")
      )
      
    )
  )
)


#_____________________________________________________________________________
## Predict -- SERVER
#_____________________________________________________________________________
predict_server <- function(input, output, session) {
  
  ## ___ Reactive expressions ___
  
  ## Read the uploaded file. Everything downstream reads this.
  uploaded_data <- reactive({
    req(input$upload)
    read_csv(input$upload$datapath, show_col_types = FALSE) |>
      mutate(across(c(date_registration, n_failed, avg_days_early,
                      total_clicks, total_days_active),
                    as.numeric))
  })
  
  ## Score once -- attach the class prediction and both probabilities.
  scored <- reactive({
    req(uploaded_data())
    data <- uploaded_data()
    data |>
      bind_cols(predict(final_fit, data)) |>
      bind_cols(predict(final_fit, data, type = "prob"))
  })
  
  ## Narrow the scored data by the three dropdowns. "All" skips that filter.
  filtered <- reactive({
    result <- scored()
    if (input$region != "All") {
      result <- result |> filter(region == input$region)
    }
    if (input$gender != "All") {
      result <- result |> filter(gender == input$gender)
    }
    if (input$age_band != "All") {
      result <- result |> filter(age_band == input$age_band)
    }
    result
  })
  
  ## ___ Outputs ___
  
  ## Table: id, prediction, and both probabilities as percentages.
  output$scored_table <- renderDT({
    filtered() |>
      transmute(
        id_student,
        Prediction  = .pred_class,
        `Success %` = .pred_Successful,   # keep as numbers
        `WF %`      = .pred_WF            # keep as numbers
      ) |>
      datatable(class = "") |>
      formatPercentage(c("Success %", "WF %"), digits = 0) |>   # DT formats for display
      formatStyle(
        "Prediction",
        backgroundColor = styleEqual(c("Successful", "WF"),
                                     c("#dce8e0", "#f0ccc2")),
        color = "#3a3f42"
      )
  })
  
  ## Score cards
  output$n_scored    <- renderText({ nrow(filtered()) })
  output$n_wf        <- renderText({ sum(filtered()$.pred_class == "WF") })
  output$n_high_risk <- renderText({ sum(filtered()$.pred_WF > input$threshold) })
  
  ## Charts
  output$risk_hist <- renderPlot({
    ggplot(filtered(), aes(x = .pred_WF)) +
      geom_histogram(binwidth = 0.05, fill = "#9EA9B9", color = "white") +
      geom_vline(xintercept = input$threshold,
                 linetype = "dashed", linewidth = 1) +
      labs(x = "Predicted probability of WF", y = "Students") +
      theme_minimal(base_size = 14)
  })
  
  output$wf_by_module <- renderPlot({
    filtered() |>
      group_by(code_module) |>
      summarise(wf_rate = mean(.pred_class == "WF")) |>
      ggplot(aes(x = code_module, y = wf_rate)) +
      geom_col(fill = "#9EA9B9") +
      labs(x = "Module", y = "Predicted WF rate") +
      theme_minimal(base_size = 14)
  })
  
  ## Download the filtered table as a CSV.
  output$download_scored <- downloadHandler(
    filename = function() "scored_students.csv",
    content  = function(file) write_csv(filtered(), file)
  )
  
}




#_______________________________________________________________________________
#_______________________________________________________________________________
#_______________________________________________________________________________
####EXAMINE####

#═════════════════════════════════════════════════════════════════════════════
#  3. EXAMINE PAGE
#
#  Examine is a single student scorer: the user builds one student with
#  controls, and clicks a button to see the model's prediction.
#
#  LAYOUT: a 4/8 split -- inputs on the LEFT (4 units), the result on the RIGHT
#  (8 units). The left column stacks three input cards plus a button.
#
#  ID: every id on this page is prefixed "ex_" so it can't collide with the
#  Predict page's ids -- all pages share one pool of input/output ids.
#═════════════════════════════════════════════════════════════════════════════

#_____________________________________________________________________________
## Examine -- UI  (finished in the previous lesson)
#_____________________________________________________________________________
examine_page <- nav_panel(
  "Examine",
  
  ## 4/8 split: inputs left, result right.
  layout_columns(
    col_widths = c(4, 8),
    
    ## ═══ LEFT: three input cards + the button ═══
    tags$div(   # tags = "this is html"; div = an empty container that groups things
      
      ## ── Card 1: Demographics ──
      card(
        card_header("Student demographics"),
        selectInput("ex_gender", "Gender:",
                    choices = c("F", "M"), selected = "M"),
        selectInput("ex_region", "Region:",
                    choices = c("East Anglian Region", "East Midlands Region",
                                "Ireland", "London Region", "North Region",
                                "North Western Region", "Scotland",
                                "South East Region", "South Region",
                                "South West Region", "Wales",
                                "West Midlands Region", "Yorkshire Region"),
                    selected = "Scotland"),
        selectInput("ex_highest_education", "Highest education:",
                    choices = c("A Level or Equivalent", "HE Qualification",
                                "Lower Than A Level", "No Formal quals",
                                "Post Graduate Qualification"),
                    selected = "A Level or Equivalent"),
        ## NAMED VECTOR: shows "10-20%" but sends the stored value "10-20"
        selectInput("ex_imd_band", "IMD band:",
                    choices = c("0-10%"   = "0-10%",  "10-20%"  = "10-20",
                                "20-30%"  = "20-30%", "30-40%"  = "30-40%",
                                "40-50%"  = "40-50%", "50-60%"  = "50-60%",
                                "60-70%"  = "60-70%", "70-80%"  = "70-80%",
                                "80-90%"  = "80-90%", "90-100%" = "90-100%"),
                    selected = "20-30%"),
        selectInput("ex_age_band", "Age band:",
                    choices = c("0-35", "35-55", "55<="), selected = "0-35"),
        selectInput("ex_disability", "Disability:",
                    choices = c("N", "Y"), selected = "N")
      ),
      
      ## ── Card 2: Course ──
      card(
        card_header("Course variables"),
        selectInput("ex_code_module", "Module:",
                    choices = c("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"),
                    selected = "BBB"),
        ## domain shown read-only, derived from the module above
        ## (the server's ex_domain output fills this slot)
        div(tags$small("Domain (set by module): "),
            textOutput("ex_domain", inline = TRUE)),
        selectInput("ex_code_presentation", "Presentation:",
                    choices = c("2013B", "2013J", "2014B", "2014J"),
                    selected = "2014J"),
        numericInput("ex_studied_credits", "Studied credits:",
                     value = 60, min = 30, max = 655),
        numericInput("ex_date_registration", "Registration day (- = before start):",
                     value = -57, min = -322, max = 167),
        numericInput("ex_num_of_prev_attempts", "Previous attempts:",
                     value = 0, min = 0, max = 6),
        numericInput("ex_module_presentation_length", "Course length (days):",
                     value = 262, min = 234, max = 269)
      ),
      
      ## ── Card 3: Engagement ──
      card(
        card_header("Engagement"),
        numericInput("ex_total_clicks", "Total clicks:",
                     value = 740, min = 1, max = 24139),
        numericInput("ex_total_days_active", "Days active:",
                     value = 47, min = 1, max = 286),
        numericInput("ex_avg_days_early", "Avg days early (- = late):",
                     value = 0, min = -187, max = 236),
        numericInput("ex_n_failed", "Assessments failed:",
                     value = 0, min = 0, max = 9)
      ),
      
      ## ── The button ──
      actionButton("run_examine", "Predict", class = "btn-primary",
                   style = "width: 100%; margin-top: 0.5rem;")
    ),
    
    ## ═══ RIGHT: the result (its slots stay blank until the server fills them) ═══
    tags$div(
      
      style = "position: sticky; top: 1rem;",   # <- pins the whole right column
      
      ## Prediction card
      card(
        max_height = 350,                          # keeps the card from stretching tall
        card_header("Prediction"),
        
        ## Predicted outcome above the value boxes (server fills ex_class)
        div(style = "margin-bottom: 1rem;",
            tags$strong("Predicted outcome: "),
            textOutput("ex_class", inline = TRUE)),
        
        p(tags$small(
          "The model's predicted probabilities of success or withdraw/failure:"
        )),
        
        ## Two probability value boxes (server fills ex_p_wf / ex_p_success)
        layout_columns(
          col_widths = c(6, 6),
          value_box(title = "Probability of WF",
                    value = textOutput("ex_p_wf"),
                    theme = value_box_theme(bg = "#f0ccc2", fg = "#3a3f42")),
          value_box(title = "Probability of Success",
                    value = textOutput("ex_p_success"),
                    theme = value_box_theme(bg = "#dce8e0", fg = "#3a3f42"))
        )
      ),
      
      ## Comparison chart card, below the prediction (server fills ex_vs_typical)
      card(
        card_header("This student vs. typical"),
        plotOutput("ex_vs_typical")
      )
      
    )
  )
)


#═════════════════════════════════════════════════════════════════════════════
#  EXAMINE SERVER -- BUILD IT ONE PIECE AT A TIME
#
#  Reactive graph we are building toward:
#
#    input$run_examine ──► examined() ──┬──► ex_p_wf
#    (the button)         (conductor)   ├──► ex_p_success
#                                       ├──► ex_class
#                                       └──► ex_vs_typical
#
#    input$ex_code_module ──► ex_domain        (live, separate, NO button)
#
#  Note the two tracks: everything off examined() waits for the button, but
#  ex_domain is live -- it reacts to the module dropdown directly. We build the
#  live one first (it is the simplest and the clearest contrast), then the
#  button-gated conductor and its outputs.
#═════════════════════════════════════════════════════════════════════════════




#═════════════════════════════════════════════════════════════════════════════
#  EXAMINE STEP 5 -- the comparison chart  (Examine server complete)
#  The last output: a bar chart comparing this student to the typical student
#  on three numeric variables. We build one small tibble per student (variable,
#  value, who), stack them with bind_rows, and facet by variable so each gets
#  its own panel and y-scale. Reads examined(), so it waits for the click too.
#═════════════════════════________
examine_server <- function(input, output, session) {
  
  ## ── Read-only labeled domain display (STEM or SS) ──
  ## Reacts to the module dropdown directly -- LIVE, not gated by the button.
  output$ex_domain <- renderText({
    # input$ex_code_module  -- the module the user picked, e.g. "CCC"
    # module_domain[["CCC"]] -- index the named vector by that string -> "STEM"
    module_domain[[input$ex_code_module]]
  })
  
  
  ## ── Score on button click only (eventReactive gates everything) ──
  examined <- eventReactive(input$run_examine, {
    
    ## ── Validate numeric inputs are in range ──
    ## If any check fails, the prediction stops and the message shows in the
    ## output slot instead of a result. between(x, lo, hi) is dplyr's range check.
    validate(
      need(between(input$ex_num_of_prev_attempts, 0, 6),
           "Previous attempts must be between 0 and 6."),
      need(between(input$ex_studied_credits, 30, 655),
           "Studied credits must be between 30 and 655."),
      need(between(input$ex_date_registration, -322, 167),
           "Registration day must be between -322 and 167."),
      need(between(input$ex_n_failed, 0, 9),
           "Assessments failed must be between 0 and 9."),
      need(between(input$ex_avg_days_early, -187, 236),
           "Average days early must be between -187 and 236."),
      need(between(input$ex_total_clicks, 1, 24139),
           "Total clicks must be between 1 and 24,139."),
      need(between(input$ex_total_days_active, 1, 286),
           "Days active must be between 1 and 286."),
      need(between(input$ex_module_presentation_length, 234, 269),
           "Course length must be between 234 and 269.")
    )
    
    ## Build the one-row student: 16 inputs + 2 required columns.
    student <- tibble(
      id_student = 999999,                                # hardcoded, ignored
      domain     = module_domain[[input$ex_code_module]], # derived, ignored
      
      code_module       = input$ex_code_module,
      code_presentation = input$ex_code_presentation,
      gender            = input$ex_gender,
      region            = input$ex_region,
      highest_education = input$ex_highest_education,
      imd_band          = input$ex_imd_band,
      age_band          = input$ex_age_band,
      disability        = input$ex_disability,
      num_of_prev_attempts       = input$ex_num_of_prev_attempts,
      studied_credits            = input$ex_studied_credits,
      date_registration          = input$ex_date_registration,
      n_failed                   = input$ex_n_failed,
      avg_days_early             = input$ex_avg_days_early,
      total_clicks               = input$ex_total_clicks,
      total_days_active          = input$ex_total_days_active,
      module_presentation_length = input$ex_module_presentation_length
    )
    
    ## Score it -- class and both probabilities.
    ## bind_cols() -- joins side by side, adds columns, matches by row position
    student |>
      bind_cols(predict(final_fit, student)) |>
      bind_cols(predict(final_fit, student, type = "prob"))
  })
  
  
  ## ___ Outputs: read examined(), so they populate only after a click ___
  
  ## Predicted class (WF / Successful)
  output$ex_class <- renderText({
    as.character(examined()$.pred_class)
  })
  
  ## P(WF) as a %
  output$ex_p_wf <- renderText({
    scales::percent(examined()$.pred_WF, accuracy = 1)
  })
  
  ## P(Successful) as a %
  output$ex_p_success <- renderText({
    scales::percent(examined()$.pred_Successful, accuracy = 1)
  })
  
  
  
  
  
  ##Now we add the faceted charts:
  ## ── This student vs. typical: faceted comparison of numeric variables ──
  output$ex_vs_typical <- renderPlot({
    scored <- examined()
    
    ## One tibble per student; each is just: variable, value, who.
    this_student <- tibble(
      variable = c("Total clicks", "Days active", "Studied credits"),
      value    = c(scored$total_clicks, scored$total_days_active, scored$studied_credits),
      who      = "This student"
    )
    
    typical <- tibble(
      variable = c("Total clicks", "Days active", "Studied credits"),
      value    = c(typical_student$total_clicks, typical_student$total_days_active,
                   typical_student$studied_credits),
      who      = "Typical"
    )
    
    ## Stack them -- both already have a who column, so plain bind_rows.
    plot_data <- bind_rows(this_student, typical)
    
    view(plot_data) ## Inspect --see the stacked tibble in the console. Remove later.
    
    ggplot(plot_data, aes(x = who, y = value, fill = who)) +
      geom_col() +
      facet_wrap(~ variable, scales = "free_y") +
      scale_fill_manual(values = c("This student" = "#9EA9B9",
                                   "Typical" = "#D4D9E0")) +
      labs(x = NULL, y = NULL, fill = NULL) +
      theme_minimal(base_size = 12)
  })
  
}




#═════════════════════════════════════════════════════════════════════════════
#  4. ABOUT PAGE
#  Static text -- no server needed.
#
#  withTags() lets us write html tags without the tags$ prefix -- inside the
#  block, strong(), div(), ul(), li(), small() are all bare.
#═════════════════════════════════════════════════════════════════════════════
about_page <- nav_panel(
  "About",
  
  withTags(
    ## A readable column -- cap the width so long text doesn't stretch edge to edge.
    div(
      style = "max-width: 800px; display: flex; flex-direction: column; gap: 1.5rem;",
      
      ## ── The question ──
      card(
        style = "border-left: 4px solid #9EA9B9;",
        card_header(
          bsicons::bs_icon("question-circle", style = "color: #9EA9B9;"),
          " Early Identification of At-Risk Students"
        ),
        p(strong("Purpose."), "The purpose of this model is as a proof of concept of early identification of students who may
          withdraw or fail a course at Open University so that intervention can occur.It is only a proof of concept
          and should be regarded as such"),
        
        p(strong("The model is offered in response to the following applied question:"),
          "Which currently enrolled students are at risk of not completing a course successfully?"),
        
        p(strong("Problem."), "Withdrawal and failure are a persistent, costly problem at the Open
          University. The OU teaches at a large scale and almost entirely
          at a distance with high rates of withdraw and failure as is common for online degree programs.
          Because students study remotely, those heading
          toward withdrawal or failure are easy to miss until the outcome is already
          recorded and it is too late to intervene."),
      ),
      
      ## ── The data ──
      card(
        style = "border-left: 4px solid #9EA9B9;",
        card_header(
          bsicons::bs_icon("database", style = "color: #9EA9B9;"),
          " The Data"
        ),
        p("Trained on a subset of the Open University Learning Analytics Dataset (OULAD)–
           roughly 32,000 student-course enrollments. One row = one student's
           registration in one course. The unit of analysis is the enrollment,
           not the student; the same person can appear more than once."),
        
        p(strong("Features")),
        ul( # ul is unordered list; li is list item
          li(strong("Demographics."), " gender, region, age band,
                highest prior education, deprivation band (IMD), disability."),
          li(strong("Course."), " module, presentation, credits
                studied, registration timing, prior attempts, course length."),
          li(strong("Engagement."), " total clicks in the online
                environment, days active, assessment timing, early failures.")
        ),
        
        p(strong("Target")), 
        ul(
          li(strong("WF"), "-- whether the enrollment ended in withdrawal
           or failure (\"WF\") versus success."))
      ),
      
      
      
      
      
      
      ## ── The model ──
      card(
        style = "border-left: 4px solid #9EA9B9;",
        card_header(
          bsicons::bs_icon("cpu", style = "color: #9EA9B9;"),
          " The Model"
        ),
        p(strong("Supervised learning. "),
          "The model was trained by supervised machine learning. It was shown
           one thousand past enrollments whose outcomes were already known —
           who withdrew or failed and who succeeded — and it learned the
           patterns that separate the two. Once trained, it applies those
           patterns to a new student it has never seen and estimates how likely
           that student is to withdraw or fail."),
        
        p(strong("Gradient-boosted trees (XGBoost). "),
          "The specific algorithm used is a gradient-boosted tree model, built with the
           tidymodels framework in R. Rather than a single cohesive model, gradient
           boosting builds many small decision trees in sequence, where each new
           tree focuses on the cases the previous ones got wrong. Stacked
           together, these hundreds of small corrections add up to a single
           strong model — which is why gradient boosting is one of the
           most reliable methods available for this kind of tabular data."),
        
        p(strong("How it was built. "),
          "The model was developed through a tidymodels supervised-learning
           workflow. The data were split into a training set and a separate test
           set, so the model could be judged on students it had never seen. On
           the training set, the data were cleaned and prepared. Rare categories
           were handled and variables were standardized as needed. The model's settings
           were tuned to find the combination that yielded the best predictive performance. 
           Throughout,care was taken to avoid ", em("leakage"), ": any information that would
           not actually be known partway through a course (such as final grades
           or whether a student ultimately un-enrolled) was withheld, so the
           model only learns from signals that would genuinely be available in
           time to act. The tuned model was then locked in and scored once
           against the held-out test set to produce the performance figures
           above."),
        
        p(strong("Performance. "),
          "On held-out data the model was not trained on, it correctly
           classifies roughly [XX]% of enrollments, with an AUC of about
           [0.XX], meaning that given one student who withdrew/failed and one
           who succeeded, the model ranks the higher-risk student correctly
           about [XX]% of the time."),
        
        
        p(strong("Probability. "),
          "The model outputs a probability, not a certainty. It reflects
           patterns in past data and can be wrong for any individual student. It
           is a tool to prompt attention and conversation, not a decision, and
           it should never be the sole basis for a judgment about a student.")
      ),
      
      ## ── The app ──
      card(
        style = "border-left: 4px solid #9EA9B9;",
        card_header(
          bsicons::bs_icon("window-stack", style = "color: #9EA9B9;"),
          " The Machine Learning Application"
        ),
        p(strong("How the model and the app fit together. "),
          "The trained model was exported from the modelling process and is loaded when the app starts.
           The app serves as a front end to that model offering a dynamic application.
           The user provides student information and the app hands it to the model. The model's probability 
           predictions are returned and displayed. This application is dynamic in that live scoring occurs
           on unseen data computing a new prediction each time."),
        
        p(strong("The Predict page to score a whole group. "),
          "Upload a CSV of students and the app scores every one at once, then
           lets you explore the results:"),
        ul(
          li(strong("Summary cards."), " how many students are in view, how
             many the model predicts will withdraw or fail, and how many exceed
             the high-risk threshold you set with the slider."),
          li(strong("Filters."), " narrow the view by region, gender, or age
             band to focus on a subgroup; every card, chart, and table updates
             to match."),
          li(strong("Charts."), " the distribution of predicted WF risk across
             the group, and the predicted WF rate broken down by course module."),
          li(strong("Table."), " each student's predicted outcome and their
             probabilities of success and of WF, which you can download as a CSV.")
        ),
        p("Read the Predict page as a triage view: it points to ",
          em("where"), " risk is concentrated -- which subgroups, which modules,
           how many students -- so attention can go where it is most needed."),
        
        p(strong("The Examine page to score one student. "),
          "Build a single hypothetical student by hand using the input controls,
           click Predict, and the app scores that one profile:"),
        ul(
          li(strong("Predicted outcome. "), "the model's call (WF or Successful)
             for the student you built. "),
          li(strong("Probabilities. "), "the chance of WF and the chance of
             success, shown side by side. The two add to 100%; above 50% WF is a
             predicted non-completion. "),
          li(strong("Comparison chart. "), "how your student's engagement and
             course load stack up against a typical student, so you can see what
             is driving the prediction. ")
        ),
        p("Read the Examine page as a what-if tool: change one input, re-run, and
           watch the probability move. It is the place to build intuition for
           ", em("which"), " factors the model responds to, and by how much."),
        
        p(strong("Built in Shiny. "),
          "The app is written in Shiny, which uses R code to create interactive
           web applications.R also trained the model that powers the
           predictive interface, so the model and the tool around it are one continuous
           workflow.")
      ),
      
      ## ── About the project ──
      card(
        style = "border-left: 4px solid #9EA9B9;",
        card_header(
          bsicons::bs_icon("mortarboard", style = "color: #9EA9B9;"),
          " About this project"
        ),
        p("This applications was built for HRED-7190 as a teaching example of an applied machine-learning
           and shiny application coding workflow ")
      )
    )
  )
  )











#═════════════════════════════════════════════════════════════════════════════
#  5. ASSEMBLE
#  page_fluid wraps everything; the theme lives here once. The navset lists the
#  page ui objects. The assembler server calls each page's server.
#═════════════════════════════════════════════════════════════════════════════

#_____________________________________________________________________________
## The app UI
#_____________________________________________________________________________
ui <- page_fluid(
  
  theme = bs_theme(
    bg      = "rgb(255, 255, 255)",
    fg      = "rgb(50, 50, 50)",
    primary = "#9EA9B9",
    success = "#687270",
    base_font        = font_google("Inter"),
    heading_font     = font_google("Poppins"),
    `border-radius`  = "0.5rem",
    `enable-shadows` = TRUE
  ),
  
  titlePanel(
    tags$span(#[!Important] If you use images, they have to live in a folder called
      #"www" in the same folder as your app script.
      tags$img(src = "ou_logo.png", height = "32px",
               style = "vertical-align: middle; margin-right: 0.5rem;"),
      "Course Completion Risk Tool "
    )
  ),
  
  navset_pill_list(
    widths = c(2, 10),
    predict_page,     ## from section 2
    examine_page,     ## from section 3
    about_page        ## from section 4
  )
)

#_____________________________________________________________________________
## The app SERVER -- calls each page's server, passing the shared
## input / output / session down to each one.
#_____________________________________________________________________________
server <- function(input, output, session) {
  predict_server(input, output, session)
  examine_server(input, output, session)
  ## about has no server -- it is static
}

#_____________________________________________________________________________
## Launch
#_____________________________________________________________________________
shinyApp(ui, server)