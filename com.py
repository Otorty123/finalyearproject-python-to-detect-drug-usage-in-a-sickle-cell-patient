import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import joblib
import logging
import mysql.connector
from mysql.connector import Error
# import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
# import seaborn as sns
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "dataset_path": "extended_synthetic_sickle_cell_data.csv",
    "model_path": "sickle_cell_detection_model.pkl",
    "scaler_path": "scaler.pkl",
    "n_samples": 5000,
    "test_size": 0.2,
    "random_state": 42,
    "flask_host": "0.0.0.0",
    "flask_port": 5000,
    "mysql_host": "localhost",
    "mysql_user": "root",
    "mysql_password": "",
    "mysql_database": "sickle_cell_data"
}

# MySQL connection function
def create_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=CONFIG["mysql_host"],
            user=CONFIG["mysql_user"],
            password=CONFIG["mysql_password"],
            database=CONFIG["mysql_database"]
        )
        if connection.is_connected():
            logger.info('Connected to MySQL database')
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

# Generate synthetic dataset
def generate_dataset(config):
    try:
        np.random.seed(config["random_state"])
        n_samples = config["n_samples"]

        ages = np.random.randint(18, 80, size=n_samples)
        genders = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.45, 0.55])
        ethnicities = np.random.choice(['African', 'Hispanic', 'Caucasian', 'Asian', 'Other'], size=n_samples,
                                       p=[0.30, 0.20, 0.35, 0.10, 0.05])
        genotype = np.random.choice(['HbSS', 'HbSC', 'HbS beta-thalassemia'], size=n_samples,
                                     p=[0.70, 0.20, 0.10])
        frequent_crises = np.random.randint(0, 20, size=n_samples)
        hospitalizations = np.random.randint(0, 10, size=n_samples)
        blood_transfusions = np.random.randint(0, 5, size=n_samples)
        drugs = np.random.choice(['Hydroxyurea', 'Penicillin', 'Folic Acid', 'L-Glutamine', 'Voxelotor'], size=n_samples)
        duration_of_usage_years = np.random.randint(1, 20, size=n_samples)
        hb_levels = np.random.normal(loc=9, scale=1.5, size=n_samples)
        wbc_counts = np.random.normal(loc=8, scale=2, size=n_samples)
        plt_counts = np.random.normal(loc=300, scale=50, size=n_samples)
        complications = np.random.choice(['None', 'Organ Damage', 'Chronic Pain', 'Infection', 'Acute Chest Syndrome'],
                                         size=n_samples, p=[0.4, 0.15, 0.2, 0.1, 0.15])
        long_term_effect = np.where(complications != 'None', 1, 0)

        data = {
            'patient_id': range(1, n_samples + 1),
            'age': ages,
            'gender': genders,
            'ethnicity': ethnicities,
            'genotype': genotype,
            'frequent_crises': frequent_crises,
            'hospitalizations': hospitalizations,
            'blood_transfusions': blood_transfusions,
            'drug_used': drugs,
            'duration_of_usage_years': duration_of_usage_years,
            'hb_levels': hb_levels,
            'wbc_counts': wbc_counts,
            'plt_counts': plt_counts,
            'complications': complications,
            'long_term_effect': long_term_effect
        }

        df = pd.DataFrame(data)
        df = feature_engineering(df)
        df.to_csv(config["dataset_path"], index=False)
        logger.info("Dataset generated successfully.")
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise

# Feature engineering
def feature_engineering(df):
    df['total_health_impact'] = df['hospitalizations'] + df['blood_transfusions']
    return df

# Cross-validation
def perform_cross_validation(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Mean cross-validation score: {np.mean(scores)}")

# Hyperparameter tuning
def perform_hyperparameter_tuning(X, y):
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.01, 0.1, 0.2]
    }

    rf_clf = RandomForestClassifier(random_state=CONFIG["random_state"])
    gb_clf = GradientBoostingClassifier(random_state=CONFIG["random_state"])
    voting_clf = VotingClassifier([('rf', rf_clf), ('gb', gb_clf)], voting='soft')

    grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_

# Train the model
def train_model(config):
    try:
        data = pd.read_csv(config["dataset_path"])
        data = pd.get_dummies(data)
        X = data.drop(columns=['patient_id', 'long_term_effect'])
        y = data['long_term_effect']

        scaler = StandardScaler()
        numerical_features = ['age', 'frequent_crises', 'hospitalizations', 'blood_transfusions', 'duration_of_usage_years',
                              'hb_levels', 'wbc_counts', 'plt_counts', 'total_health_impact']
        X[numerical_features] = scaler.fit_transform(X[numerical_features])
        joblib.dump(scaler, config["scaler_path"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

        # Hyperparameter Tuning
        best_model = perform_hyperparameter_tuning(X_train, y_train)

        # Cross-Validation
        perform_cross_validation(best_model, X_train, y_train)

        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info(f"\nClassification Report:\n{class_report}")

        joblib.dump(best_model, config["model_path"])
        logger.info("Model trained and saved successfully.")
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

# Flask app setup
app = Flask(__name__)


def deploy_model(config):
    try:
        
        # Flask route to detect sickle cell disease
        @app.route('/detect', methods=['POST'])
        def detect_sickle_cell_disease():
            try:
                data = request.get_json()
                input_data = pd.DataFrame([data])
                input_data = pd.get_dummies(input_data)
                model_features = model.feature_names_in_
                input_data = input_data.reindex(columns=model_features, fill_value=0)
                numerical_features = ['age', 'frequent_crises', 'hospitalizations', 'blood_transfusions', 'duration_of_usage_years',
                                    'hb_levels', 'wbc_counts', 'plt_counts', 'total_health_impact']
                input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                prediction = model.predict(input_data)[0]
                outcome = "positive" if prediction else "negative"
                return jsonify({'long_term_effect_detected': bool(prediction), 'outcome': outcome})
            except Exception as e:
                logger.error(f"Error in detection endpoint: {e}")
                return jsonify({'error': str(e)}), 400
            
            
        @app.route('/retrieve', methods=['GET'])
        def retrieve_patients():
            try:
                connection = create_mysql_connection()
                if connection:
                    cursor = connection.cursor(dictionary=True)
                    sql = "SELECT * FROM patient_data"
                    cursor.execute(sql)
                    records = cursor.fetchall()
                    return jsonify(records)
                else:
                    return jsonify({'error': 'Could not connect to the database'}), 500
            except Exception as e:
                logger.error(f"Error retrieving patients: {e}")
                return jsonify({'error': str(e)}), 400
            finally:
                if connection and connection.is_connected():
                    cursor.close()
                    connection.close()
                    logger.info('MySQL connection closed')

        # Endpoint to add a new patient record
        @app.route('/add', methods=['POST'])
        def add_patient():
            try:
                data = request.get_json()
                connection = create_mysql_connection()
                if connection:
                    cursor = connection.cursor()
                    sql = """
                    INSERT INTO patient_data (age, gender, ethnicity, genotype, frequent_crises, hospitalizations,
                                            blood_transfusions, drug_used, duration_of_usage_years, hb_levels,
                                            wbc_counts, plt_counts, complications, long_term_effect)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (
                        data.get('age'),
                        data.get('gender'),
                        data.get('ethnicity'),
                        data.get('genotype'),
                        data.get('frequent_crises'),
                        data.get('hospitalizations'),
                        data.get('blood_transfusions'),
                        data.get('drug_used'),
                        data.get('duration_of_usage_years'),
                        data.get('hb_levels'),
                        data.get('wbc_counts'),
                        data.get('plt_counts'),
                        data.get('complications'),
                        int(data.get('long_term_effect'))
                    )
                    cursor.execute(sql, values)
                    connection.commit()
                    logger.info("New patient record added successfully.")
                    return jsonify({'message': 'New patient record added successfully.'}), 200
                else:
                    return jsonify({'error': 'Could not connect to the database'}), 500
            except Exception as e:
                logger.error(f"Error adding new patient record: {e}")
                return jsonify({'error': str(e)}), 400
            finally:
                if connection and connection.is_connected():
                    cursor.close()
                    connection.close()
                    logger.info('MySQL connection closed')

        # Endpoint to update a patient record
        @app.route('/update/<int:id>', methods=['PUT'])
        def update_patient(id):
            try:
                data = request.get_json()
                connection = create_mysql_connection()
                if connection:
                    cursor = connection.cursor()
                    sql = """
                    UPDATE patient_data
                    SET age=%s, gender=%s, ethnicity=%s, genotype=%s, frequent_crises=%s,
                        hospitalizations=%s, blood_transfusions=%s, drug_used=%s,
                        duration_of_usage_years=%s, hb_levels=%s, wbc_counts=%s, plt_counts=%s,
                        complications=%s, long_term_effect=%s
                    WHERE id=%s
                    """
                    values = (
                        data.get('age'),
                        data.get('gender'),
                        data.get('ethnicity'),
                        data.get('genotype'),
                        data.get('frequent_crises'),
                        data.get('hospitalizations'),
                        data.get('blood_transfusions'),
                        data.get('drug_used'),
                        data.get('duration_of_usage_years'),
                        data.get('hb_levels'),
                        data.get('wbc_counts'),
                        data.get('plt_counts'),
                        data.get('complications'),
                        int(data.get('long_term_effect')),
                        id
                    )
                    cursor.execute(sql, values)
                    connection.commit()
                    logger.info(f"Patient record with ID {id} updated successfully.")
                    return jsonify({'message': f'Patient record with ID {id} updated successfully.'}), 200
                else:
                    return jsonify({'error': 'Could not connect to the database'}), 500
            except Exception as e:
                logger.error(f"Error updating patient record: {e}")
                return jsonify({'error': str(e)}), 400
            finally:
                if connection and connection.is_connected():
                    cursor.close()
                    connection.close()
                    logger.info('MySQL connection closed')

        # Endpoint to delete a patient record
        @app.route('/delete/<int:id>', methods=['DELETE'])
        def delete_patient(id):
            try:
                connection = create_mysql_connection()
                if connection:
                    cursor = connection.cursor()
                    sql = "DELETE FROM patient_data WHERE id=%s"
                    cursor.execute(sql, (id,))
                    connection.commit()
                    logger.info(f"Patient record with ID {id} deleted successfully.")
                    return jsonify({'message': f'Patient record with ID {id} deleted successfully.'}), 200
                else:
                    return jsonify({'error': 'Could not connect to the database'}), 500
            except Exception as e:
                logger.error(f"Error deleting patient record: {e}")
                return jsonify({'error': str(e)}), 400
            finally:
                if connection and connection.is_connected():
                    cursor.close()
                    connection.close()
                    logger.info('MySQL connection closed')

        logger.info(f"Starting Flask server on http://{config['flask_host']}:{config['flask_port']}")
        # app.run(debug=True, host=config["flask_host"], port=config["flask_port"])

    except Exception as e:
                logger.error(f"Error deploying model with Flask: {e}")
                raise


# Function to create GUI layout
root = tk.Tk()
def create_gui_layout():
    
    try:
        root.configure(background='blue')
        root.title("Sickle Cell Disease Detection")
        window_width = 1200
        window_height = 600
        root.geometry(f"{window_width}x{window_height}")
        message = tk.Label(root, text="DETECTION OF LONG TERM EFFECT OF DRUG USAGE IN \n  SICKLE CELL PATIENT", bg="green", fg="white", width=50, height=2, font=('times', 20, 'bold'))
        message.place(x=250, y=20)

        frame = ttk.Frame(root)
        frame.grid(row=0, column=0)
        frame.place(x=450, y=100)

        labels = ["Age", "Gender", "Ethnicity", "Genotype", "Frequent Crises", "Hospitalizations", "Blood Transfusions",
                  "Drug Used", "Duration of Usage (Years)", "Hemoglobin Levels", "WBC Counts", "Platelet Counts", "Complications"]

        row = 0
        entries = {}

        for label in labels:
            ttk.Label(frame, text=label).grid(row=row, column=0, padx=5, pady=5)
            if label in ["Gender", "Ethnicity", "Genotype", "Drug Used", "Complications"]:
                var = tk.StringVar()
                combobox = ttk.Combobox(frame, textvariable=var)
                combobox.grid(row=row, column=1, padx=5, pady=5)
                entries[label] = combobox
                if label == "Gender":
                    combobox["values"] = ["Male", "Female"]
                    gender_var = var
                elif label == "Ethnicity":
                    combobox["values"] = ["African", "Hispanic", "Caucasian", "Asian", "Other"]
                    ethnicity_var = var
                elif label == "Genotype":
                    combobox["values"] = ["HbSS", "HbSC", "HbS beta-thalassemia"]
                    genotype_var = var
                elif label == "Drug Used":
                    combobox["values"] = ["Hydroxyurea", "Penicillin", "Folic Acid", "L-Glutamine", "Voxelotor"]
                    drug_used_var = var
                elif label == "Complications":
                    combobox["values"] = ["None", "Acute Chest Syndrome", "Chronic Pain", "Infection", "Organ Damage"]
                    complications_var = var
            else:
                entry = ttk.Entry(frame)
                entry.grid(row=row, column=1, padx=5, pady=5)
                entries[label] = entry
            row += 1

        # Add result label
        result_label = ttk.Label(frame, text="Detection Result: ", font=("Helvetica", 16))
        result_label.grid(row=row, column=0, columnspan=2, pady=10)

        # Define entries for easy access
        age_entry = entries["Age"]
        gender_var = tk.StringVar()
        entries["Gender"].config(textvariable=gender_var)
        ethnicity_var = tk.StringVar()
        entries["Ethnicity"].config(textvariable=ethnicity_var)
        genotype_var = tk.StringVar()
        entries["Genotype"].config(textvariable=genotype_var)
        frequent_crises_entry = entries["Frequent Crises"]
        hospitalizations_entry = entries["Hospitalizations"]
        blood_transfusions_entry = entries["Blood Transfusions"]
        drug_used_var = tk.StringVar()
        entries["Drug Used"].config(textvariable=drug_used_var)
        duration_of_usage_years_entry = entries["Duration of Usage (Years)"]
        hb_levels_entry = entries["Hemoglobin Levels"]
        wbc_counts_entry = entries["WBC Counts"]
        plt_counts_entry = entries["Platelet Counts"]
        complications_var = tk.StringVar()
        entries["Complications"].config(textvariable=complications_var)


        
        def add_patient():
            patient_data = {
                'age': int(age_entry.get()),
                'gender': gender_var.get(),
                'ethnicity': ethnicity_var.get(),
                'genotype': genotype_var.get(),
                'frequent_crises': int(frequent_crises_entry.get()),
                'hospitalizations': int(hospitalizations_entry.get()),
                'blood_transfusions': int(blood_transfusions_entry.get()),
                'drug_used': drug_used_var.get(),
                'duration_of_usage_years': int(duration_of_usage_years_entry.get()),
                'hb_levels': float(hb_levels_entry.get()),
                'wbc_counts': float(wbc_counts_entry.get()),
                'plt_counts': float(plt_counts_entry.get()),
                'complications': complications_var.get()
            }

            # Insert the new patient data into the database
            connection = create_mysql_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    query = """INSERT INTO patient_data (age, gender, ethnicity, genotype, frequent_crises, hospitalizations, blood_transfusions,
                                drug_used, duration_of_usage_years, hb_levels, wbc_counts, plt_counts, complications)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
                    cursor.execute(query, tuple(patient_data.values()))
                    connection.commit()
                    logger.info("New patient record added successfully.")
                except Error as e:
                    logger.error(f"Error inserting patient data: {e}")
                    messagebox.showerror("Error", "Error adding patient.")
                finally:
                    cursor.close()
                    connection.close()
        # Function to detect disease using Flask API
        def detect_disease():
            try:
                url = 'http://127.0.0.1:5000/detect'  # Update URL as per your Flask server setup
                data = {
                    "age": int(age_entry.get()),
                    "gender": gender_var.get(),
                    "ethnicity": ethnicity_var.get(),
                    "genotype": genotype_var.get(),
                    "frequent_crises": int(frequent_crises_entry.get()),
                    "hospitalizations": int(hospitalizations_entry.get()),
                    "blood_transfusions": int(blood_transfusions_entry.get()),
                    "drug_used": drug_used_var.get(),
                    "duration_of_usage_years": int(duration_of_usage_years_entry.get()),
                    "hb_levels": float(hb_levels_entry.get()),
                    "wbc_counts": float(wbc_counts_entry.get()),
                    "plt_counts": float(plt_counts_entry.get()),
                    "complications": complications_var.get()
                }

                response = requests.post(url, json=data)
                if response.status_code == 200:
                    result = response.json()
                    outcome = "positive" if result['long_term_effect_detected'] else "negative"
                    result_label.config(text=f"Detection Result: {outcome}", foreground="green")
                else:
                    result_label.config(text="Detection failed", foreground="red")
                    messagebox.showerror("Error", f"Detection failed: {response.text}")
            except Exception as e:
                logger.error(f"Error in disease detection: {e}")
                messagebox.showerror("Error", f"An error occurred in disease detection: {e}")

        def add_and_detect():
            add_patient()
            detect_disease()


        # Function to clear form
        def clear_form():
            try:
                age_entry.delete(0, tk.END)
                gender_var.set("")
                ethnicity_var.set("")
                genotype_var.set("")
                frequent_crises_entry.delete(0, tk.END)
                hospitalizations_entry.delete(0, tk.END)
                blood_transfusions_entry.delete(0, tk.END)
                drug_used_var.set("")
                duration_of_usage_years_entry.delete(0, tk.END)
                hb_levels_entry.delete(0, tk.END)
                wbc_counts_entry.delete(0, tk.END)
                plt_counts_entry.delete(0, tk.END)
                complications_var.set("")
                result_label.config(text="Detection Result: ", foreground="black")
            except Exception as e:
                logger.error(f"Error clearing form: {e}")

        # Function to save data to a CSV file
        def save_data():
            try:
                data = {
                    "Age": int(age_entry.get()),
                    "Gender": gender_var.get(),
                    "Ethnicity": ethnicity_var.get(),
                    "Genotype": genotype_var.get(),
                    "Frequent Crises": int(frequent_crises_entry.get()),
                    "Hospitalizations": int(hospitalizations_entry.get()),
                    "Blood Transfusions": int(blood_transfusions_entry.get()),
                    "Drug Used": drug_used_var.get(),
                    "Duration of Usage (Years)": int(duration_of_usage_years_entry.get()),
                    "Hemoglobin Levels": float(hb_levels_entry.get()),
                    "WBC Counts": float(wbc_counts_entry.get()),
                    "Platelet Counts": float(plt_counts_entry.get()),
                    "Complications": complications_var.get()
                }

                df = pd.DataFrame([data])
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    df.to_csv(file_path, index=False)
                    messagebox.showinfo("Success", "Data saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving data: {e}")

        # Function to load data from a CSV file into the form
        def load_data():
            try:
                file_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        data = df.iloc[0].to_dict()
                        for key, value in data.items():
                            if key in entries:
                                if isinstance(entries[key], tk.Entry):
                                    entries[key].delete(0, tk.END)
                                    entries[key].insert(0, str(value))
                                elif isinstance(entries[key], ttk.Combobox):
                                    entries[key].set(str(value))
                        messagebox.showinfo("Success", "Data loaded successfully!")
                    else:
                        messagebox.showwarning("Warning", "No data found in the selected file.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while loading data: {e}")


        # Add buttons
        ttk.Button(frame, text="Detect", command=add_and_detect).grid(row=row + 1, column=0, pady=5, padx=20)
        ttk.Button(frame, text="Clear", command=clear_form).grid(row=row + 1, column=1, pady=5, padx=20)
        ttk.Button(frame, text="Save Data", command=save_data).grid(row=row + 2, column=0, pady=5, padx=20)
        ttk.Button(frame, text="Load Data", command=load_data).grid(row=row + 2, column=1, pady=5, padx=20)
        root.mainloop()

    except Exception as e:
        logger.error(f"Error in GUI creation: {e}")
        messagebox.showerror("Error", f"An error occurred in GUI creation: {e}")

        # Add padding to all child widgets
        for child in frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

 # Start Flask app
def start_flask_app():
    app.run(host=CONFIG["flask_host"], port=CONFIG["flask_port"])

    # Main function to start GUI and Flask
if __name__ == '__main__':
    try:
        if not os.path.exists(CONFIG["dataset_path"]):
            generate_dataset(CONFIG)
        train_model(CONFIG)
        deploy_model(CONFIG)
        model = joblib.load(CONFIG["model_path"])
        scaler = joblib.load(CONFIG["scaler_path"])

             # Start Flask server in a separate thread
        from threading import Thread
        flask_thread = Thread(target=start_flask_app)
        flask_thread.daemon = True
        flask_thread.start()

        gui = create_gui_layout()
        gui.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
