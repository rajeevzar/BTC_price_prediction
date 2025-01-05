# Bitcoin Price Forecasting Using LSTM

## Description

This project aims to analyze and predict Bitcoin prices using historical data and a Long Short-Term Memory (LSTM) model. The script includes data preprocessing, visualization, and model training and evaluation. The primary goal is to forecast Bitcoin prices and compare the model's predictions with actual values.

The data is sourced from the [Bitcoin Historical Data Kaggle dataset](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data). Predictions are visualized using interactive Plotly plots and Matplotlib subplots.

---

## Features

1. **Data Preprocessing:**
   - Data cleaning and handling missing values.
   - Calculating the mean price from open and close prices.
   - Aggregating data at hourly and weekly intervals.

2. **Visualization:**
   - Plotly interactive time series plots with range sliders.
   - Dual-panel Matplotlib plots for full dataset visualization and zoomed-in view.

3. **Forecasting Model:**
   - Long Short-Term Memory (LSTM) neural network for time series forecasting.
   - Training and validation split with loss curve visualization.

4. **Performance Metrics:**
   - Mean Squared Error (MSE).
   - Mean Absolute Error (MAE).

---

## Libraries Used

- **Data Manipulation:**
  - `numpy`
  - `pandas`

- **Visualization:**
  - `matplotlib`
  - `seaborn`
  - `plotly`

- **Machine Learning:**
  - `sklearn`
  - `keras` (LSTM, Dense layers)

- **Utilities:**
  - `datetime`
  - `pytz`
  - `gc`

---

## How to Use

### 1. Data Preprocessing
- Load the Bitcoin dataset (`btcusd_1-min_data.csv`) and remove missing values.
- Convert timestamps to UTC and aggregate data into hourly intervals.
- Split the dataset into training and testing subsets, ensuring a valid date range.

### 2. Visualization
- Use Plotly to create interactive time series plots with a range slider.
- Generate Matplotlib subplots to visualize the full dataset and zoomed-in views.

### 3. Model Training
- Scale the data using MinMaxScaler.
- Prepare the data for LSTM input with appropriate 3D reshaping.
- Train the LSTM model with 128 units and a dropout layer for regularization.
- Early stopping is used to prevent overfitting.

### 4. Prediction
- Scale the test set using the same scaler as the training set.
- Predict Bitcoin prices using the trained LSTM model.
- Inverse-transform the scaled predictions for comparison with actual prices.

### 5. Evaluation
- Calculate and print the MSE and MAE between the predicted and actual prices.
- Visualize the loss curve for training and validation.

---

## Example Outputs

### 1. Interactive Plotly Visualization
- Mean price vs. predicted price with a range slider.

### 2. Matplotlib Subplots
- Full dataset visualization and zoomed-in prediction view.

### 3. Loss Curve
- Training and validation loss across epochs.

### 4. Evaluation Metrics
- Mean Squared Error (MSE).
- Mean Absolute Error (MAE).

---

## Results
- **MSE**: [9086683.36876058]
- **MAE**: [2499.116663607775]

---

## Dependencies

Install the required Python libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn keras
