##_____________________________________________
##Predict page of the App
##_____________________________________________

library(shiny)
library(bslib)
library(tidyverse)
library(tidymodels)
library(DT)
library(xgboost) #difference bt this upload/site app and the local file


## The trained model. This path must point to wherever you saved it.
final_fit <- readRDS("final_fit_gbm.rds")


# The app that will remain unchanged as much as possible while we work on the server:
ui <- page_fluid(
  
  theme = bs_theme(
    bg      = "rgb(255, 255, 255)",  # page background
    fg      = "rgb(50, 50, 50)",     # page foreground (main text)
    primary = "#9EA9B9",             # accent: links, buttons, active pill
    success = "#687270",             # green "good" state
    `border-radius`  = "0.5rem",     # corner rounding
    `enable-shadows` = TRUE          # subtle depth
  ),
  
  titlePanel("OULAD Explorer"),
  
  navset_pill_list( #Splits the page into two parts (nav list and main panel)
    widths = c(2, 10),
    
    ## ___ Page 1: Predict _______________________________________________
    
    nav_panel(
      "Predict",
      
      ## Upload sits at the top, full width, above everything else.
      card(
        card_header("Load data"),
        fileInput("upload", "Upload a student CSV:", accept = ".csv")
      ),
      
      ## Below the upload: filter panel on the left, content on the right.
      layout_columns(
        col_widths = c(3, 9),
        
        ## ___ Filter panel (left) ______________________________________
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
        
        ## ___ Content Page (right) __________________________________________
        
        
        ## Score cards on top, charts below. Wrapped in a div so the two
        ## rows stack within the 9-unit column.
        tags$div( #tags is how we tell shiny to use raw HTML.
          
          
          ##TOP_______
          ## Three empty score cards
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
                      p("of students in view")),# extra text below the value
            
            value_box(title = "High risk",       
                      value = textOutput("n_high_risk"),
                      showcase = bsicons::bs_icon("exclamation-octagon"),
                      theme    = value_box_theme(bg = "#e8b3a6", fg = "#3a3f42"),
                      p("above the threshold"))
          ),
          
          ##MIDDLE______
          ## Two empty chart cards
          layout_columns(
            col_widths = c(6, 6),
            card(card_header("Risk distribution"),
                 plotOutput("risk_hist")), #[IMPORTANT!] plots are added; were blank
            card(card_header("WF rate by module"),
                 plotOutput("wf_by_module"))
          ),
          
          ##BOTTOM_____
          ## Table card
          card(
            card_header("Scored students",
                        downloadButton("download_scored", "Download CSV",
                                       class = "btn-sm")),
            DTOutput("scored_table")
          )
          
        )
        
      )
    ),
    
    ## ___ Page 2: Examine (blank stub) __________________________________
    nav_panel("Examine"),
    
    ## ___ Page 3: About (blank stub) ____________________________________
    nav_panel("About")
    
  )
)
























#_____________________________________________________________________________
#_____________________________________________________________________________
##The Server_______________________________________________________________
server <- function(input, output, session) {
  
  
  #_____________________________________________________________________________
  ##__Reactive Expressions______________________________________________________
  #require the data correct it: uploaded_data() RE.
  uploaded_data <- reactive({
    req(input$upload)
    read_csv(input$upload$datapath, show_col_types = FALSE) |>
      mutate(across(c(date_registration, n_failed, avg_days_early,
                      total_clicks, total_days_active),
                    as.numeric))
  })
  
  
  
  #calculate the score: scored() RE.__________________________________________
  scored <- reactive({
    req(uploaded_data())
    data <- uploaded_data()
    data |>
      bind_cols(predict(final_fit, data)) |>
      bind_cols(predict(final_fit, data, type = "prob"))
  })
  
  
  
  ## filtered() RE.___________________________________________________________
  ## Narrow the scored data by the three dropdowns. "All" skips that filter.
  filtered <- reactive({
    result <- scored() #result is the scored data from above
    
    #if the region is not ALL, then use the input$region value
    #ALL essentially means 'dont narrow by region', skipping the filter.
    if (input$region != "All") {
      
      #  'result <- result' updates the values handed down from RE scored()
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
  
  
  
  #_____________________________________________________________________________
  #Outputs______________________________________________________________________
  #_____________________________________________________________________________
  
  #Table Output___________________________________________________________
  #This is an Output; creates the table; Nothing downstream reads it.
  output$scored_table <- renderDT({
    filtered() |>     # read the filtered data (calls the reactive exp.)
      transmute(
        id_student,
        Prediction = .pred_class,
        `Success %` = scales::percent(.pred_Successful, accuracy = 1),
        `WF %`      = scales::percent(.pred_WF,          accuracy = 1)
      ) |>
      
      #hand to datatable
      datatable(class = "") |>
      formatStyle(
        "Prediction",
        backgroundColor = styleEqual(c("Successful", "WF"),
                                     c("#dce8e0", "#f0ccc2")),
        color = "#3a3f42"
      )
  })
  
  
  
  
  
  ##Value Box Outputs_________________________________________________________
  
  #Students Scored  -- How many rows?
  output$n_scored <- renderText({
    nrow(filtered())
  })
  
  #Predicted WF -- How many does the model predict WF?
  output$n_wf <- renderText({
    sum(filtered()$.pred_class == "WF")
  })
  
  
  #High Risk - How many exceed the threshold slider?
  output$n_high_risk <- renderText({
    sum(filtered()$.pred_WF > input$threshold)
  })
  
  
  
  
  
  ##Graph Outputs____________________________________________________________
  ##histogram output
  output$risk_hist <- renderPlot({
    ggplot(filtered(), aes(x = .pred_WF)) +
      geom_histogram(binwidth = 0.05, fill = "#9EA9B9", color = "white") +
      geom_vline(xintercept = input$threshold, #[IMPORTANT!] Reacts to the slider.
                 linetype = "dashed", linewidth = 1) +
      labs(x = "Predicted probability of WF", y = "Students") +
      theme_minimal(base_size = 14)
  })
  
  
  
  
  
  
  ##Scatter plot output________________________________________________________
  output$wf_by_module <- renderPlot({
    filtered() |>
      group_by(code_module) |>
      summarise(wf_rate = mean(.pred_class == "WF")) |>
      ggplot(aes(x = code_module, y = wf_rate)) +
      geom_col(fill = "#9EA9B9") +
      labs(x = "Module", y = "Predicted WF rate") +
      theme_minimal(base_size = 14)
  })
  
  
  
  
  
  
  ## ___ Download ________________________________________________________
  output$download_scored <- downloadHandler(
    filename = function() "scored_students.csv",
    content  = function(file) write_csv(filtered(), file)
  )
  
}


shinyApp(ui, server)