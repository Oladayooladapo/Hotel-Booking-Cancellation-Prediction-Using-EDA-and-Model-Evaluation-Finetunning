# Hotel-Booking-Cancellation-Prediction-Using-EDA-and-Model-Evaluation-Finetunning
This project aims to develop a predictive model to identify hotel bookings at risk of cancellation. By leveraging historical booking data, the model provides actionable insights to Hotel Haven, a luxury hotel chain, enabling them to optimise resource allocation, reduce revenue loss from cancellations, and enhance overall customer satisfaction.

## Project Overview

This project aims to develop a predictive model to identify hotel bookings at risk of cancellation. By leveraging historical booking data, the model provides actionable insights to Hotel Haven, a luxury hotel chain, enabling them to optimise resource allocation, reduce revenue loss from cancellations, and enhance overall customer satisfaction.

The core objective is to predict the `booking status` (Cancelled or Not_Cancelled) using various features related to customer behaviour, booking details, and stay characteristics.

## Business Problem

Hotel Haven faces significant challenges due to high cancellation rates, which lead to:
- Lost revenue from unfulfilled bookings.
- Inefficient resource allocation (e.g., staffing, room availability).
- Difficulty in strategic planning and inventory management.

The existing system lacks the foresight to understand *why* customers cancel, hindering proactive measures. This project addresses this gap by building a data-driven predictive solution.

## Project Goal

To build and evaluate a robust machine learning model that accurately predicts whether a hotel booking will be cancelled or not, providing Hotel Haven with the insights needed to implement targeted strategies for cancellation reduction and improved operational efficiency.

## Dataset

The dataset used for this project is `booking.csv`, containing historical hotel booking information.

**Key Columns:**
- `Booking_ID`: Unique identifier for each booking. (Dropped during preprocessing)
- `number of adults`: Number of adults in the booking.
- `number of children`: Number of children in the booking.
- `number of weekend nights`: Number of weekend nights included in the booking.
- `number of week nights`: Number of week nights included in the booking.
- `type of meal`: Meal plan selection (e.g., Meal Plan 1, Not Selected).
- `car parking space`: Whether the booking includes parking (0: No, 1: Yes).
- `room type`: Type of room booked (e.g., Room_Type 1, Room_Type 2).
- `lead time`: Time between the reservation and check-in date (in days).
- `market segment type`: Booking channel (e.g., Online, Offline).
- `repeated`: Whether the booking is from a repeat customer (0: No, 1: Yes).
- `P-C`: Number of previous cancellations by the customer.
- `P-not-C`: Number of previous non-cancelled bookings by the customer.
- `average price`: Average price of the booking per night.
- `special requests`: Number of special requests made by the customer.
- `date of reservation`: Date the reservation was made.
- `booking status`: Status of the booking (Cancelled, Not_Canceled) - **Target Variable**.

## Project Workflow & Methodology

This project followed a standard CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, encompassing the following steps:

### 1. Data Understanding
- **Loaded essential libraries:** `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`.
- **Initial Data Inspection:**
    - Checked dataset dimensions: 36285 rows, 17 columns.
    - Verified data types and identified categorical vs. numerical features.
    - Confirmed no missing values across all columns.
    - Examined the distribution of the target variable (`booking status`): `Not_Canceled` (24396), `Cancelled` (11889).

### 2. Data Cleaning & Preprocessing
- **Handling Duplicates:** Identified and removed 10276 duplicate rows, resulting in a cleaned dataset of 26009 unique bookings.
- **Feature Exclusion:** Removed `Booking_ID` as it's a unique identifier and not relevant for prediction.
- **Outlier Treatment:**
    - Analyzed `Lead Time` and `Average Price` for outliers using statistical summaries and box plots.
    - Outliers were addressed using the Interquartile Range (IQR) method to ensure robust model performance.
- **Date Handling:** Removed 35 rows corresponding to leap year booking dates to avoid potential data inconsistencies.

### 3. Feature Engineering
- Created new features to capture additional patterns and enhance model predictive power:
    - `month of reservation`: Extracted from `date of reservation`.
    - `day of the week`: Extracted from `date of reservation`.
    - `total nights of stay`: Calculated as `number of weekend nights` + `number of week nights`.

### 4. Exploratory Data Analysis (EDA)
Comprehensive EDA was performed to uncover patterns, relationships, and key insights within the data.

- **Univariate Analysis:**
    - Visualized distributions of numerical features (`Lead Time`, `Average Price`) to understand their spread and identify anomalies.
- **Bivariate Analysis:**
    - Explored relationships between numerical and categorical features using bar plots.
    - **Key Insights Derived:**
        - **Cancellation Drivers:** Bookings with **longer lead times**, **higher average prices**, and a **greater number of special requests** were significantly more likely to be canceled.
        - **Meal Plan Impact:** "Meal Plan 2" bookings tend to have longer lead times, longer stays, and higher average prices, suggesting a premium customer segment. "Meal Plan 3" is associated with last-minute, shorter, and budget-conscious bookings.
        - **Room Type Preferences:** "Room Type 2" and "Room Type 4" are premium room types, booked well in advance, for longer stays, and with more special requests.
        - **Market Segment Behaviour:** "Online" and "Offline" segments dominate leisure travel with longer lead times and more special requests. "Corporate" and "Aviation" represent business-oriented, often last-minute, bookings.
- **Multivariate Analysis (Correlation Heatmap):**
    - Visualized correlations between numerical features.
    - Identified strong positive correlations (e.g., `number of adults` & `average price`, `repeated` & `P-not-C`).
    - Noted a strong negative correlation between `Reservation Weekday` and `Reservation Month`, indicating seasonal booking patterns for specific days.

### 5. Model Development
- **Data Preparation for ML:**
    - **Encoding:** Categorical features (`type of meal`, `room type`, `market segment type`, `month of reservation`, `day of the week`) were transformed into numerical representations using `LabelEncoder`.
    - **Scaling:** All numerical features were scaled using `StandardScaler` to ensure no single feature dominates the model due to its scale.
    - **Feature and Target Split:** The dataset was split into features (X) and the target variable (`booking status`, y).
- **Handling Class Imbalance:**
    - The dataset exhibited class imbalance (more `Not_Canceled` than `Cancelled` bookings).
    - The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to balance the classes, preventing the model from being biased towards the majority class.
- **Data Splitting:** The dataset was split into training and testing sets (e.g., 80% training, 20% testing) to evaluate model performance on unseen data.
- **Model Selection & Training:**
    - A range of classification algorithms was trained and evaluated:
        - Logistic Regression
        - Random Forest Classifier
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNeighborsClassifier)
        - Decision Tree Classifier
        - Gradient Boosting Classifier
        - AdaBoost Classifier
        - XGBoost Classifier

### 6. Model Evaluation & Fine-Tuning
- **Initial Evaluation:** Models were evaluated using:
    - **Accuracy Score:** Overall correctness of predictions.
    - **Classification Report:** Precision, recall, and F1-score for each class.
    - **Confusion Matrix:** Detailed breakdown of true positives, true negatives, false positives, and false negatives.
- **Initial Performance:** XGBoost Classifier and Random Forest Classifier emerged as the top performers with an initial accuracy of 89%.
- **Feature Importance Experiment:** An attempt to improve accuracy by selecting the top 5 most important features led to a slight drop in accuracy for both Random Forest (86%) and XGBoost (84%). This indicated that more features were beneficial.
- **Hyperparameter Tuning (GridSearchCV):**
    - To further optimize the best-performing models, `GridSearchCV` was used to search for the optimal combination of hyperparameters systematically.
    - **Final Model Performance:**
        - **Random Forest Classifier:** 89% Accuracy, 0.95 AUC Score.
        - **XGBoost Classifier:** 89% Accuracy, 0.96 AUC Score.

## Conclusion & Recommendations for Hotel Haven

Based on the comprehensive analysis and model development:

- **XGBoost Classifier** is the recommended model for predicting booking cancellations, demonstrating strong performance with **89% accuracy** and an **AUC score of 0.96**.

**Actionable Recommendations:**

1.  **Identify High-Risk Bookings:** Implement the XGBoost model to flag bookings with characteristics like:
    * **Long lead times:** These are the most significant indicators of cancellation risk.
    * **Higher average prices:** More expensive bookings are more prone to cancellation.
    * **Multiple special requests:** Guests with specific needs might be more likely to cancel if their expectations are not met or if alternative options arise.
    * **Longer planned stays:** Bookings spanning more nights show a higher cancellation likelihood.
2.  **Proactive Engagement Strategies:**
    * For high-risk bookings, consider sending personalized reminders, offering flexible rebooking options, or providing small incentives (e.g., complimentary welcome drink, early check-in) closer to the check-in date.
    * For bookings with many special requests, proactively confirm their feasibility to reduce uncertainty and potential cancellations.
3.  **Dynamic Pricing & Inventory Management:**
    * Adjust pricing strategies for bookings made with very long lead times, potentially offering non-refundable rates with slight discounts to secure commitment.
    * Optimize room inventory based on predicted cancellation rates for different room types and market segments.
4.  **Targeted Marketing:**
    * Leverage insights from meal plan and room type analysis to tailor promotions. For instance, promote "Meal Plan 2" and premium room types to guests who plan ahead, and offer last-minute deals for "Meal Plan 3."

## Future Work

To further enhance the model and derive deeper insights:

-   **Explore Different Encoding Techniques:** Experiment with `OneHotEncoder` or `TargetEncoder` for categorical features to see if it improves model performance.
-   **Alternative Scaling Methods:** Investigate other scaling techniques beyond `StandardScaler` (e.g., `MinMaxScaler`, `RobustScaler`) to handle different data distributions.
-   **Advanced Outlier Treatment:** Refine outlier detection and treatment methods for numerical features to potentially capture more nuanced patterns.
-   **Ensemble Methods & Deep Learning:** Explore more complex ensemble techniques or neural networks for potentially higher accuracy, especially with larger datasets.
-   **External Data Integration:** Incorporate external factors such as:
    -   Local events or holidays.
    -   Weather forecasts.
    -   Competitor pricing and availability.
-   **Time-Series Analysis:** If booking data includes a time component, consider time-series models to predict future booking trends and seasonality more accurately.


## Files in this Repository

-   `ML Capstone Project-Con Env.ipynb`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, feature engineering, model training, and evaluation.
-   `booking.csv`: The raw dataset used for the project.
-   `README.md`: This file.
