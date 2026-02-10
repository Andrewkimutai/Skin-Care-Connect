import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import hashlib  # For password hashing
import os
import uuid  # For unique filenames
from datetime import datetime
# Import backend functions
import sys

sys.path.append('.')  # Adjust path if necessary
from database import init_db, get_user_by_username, create_user, save_prediction_result, book_appointment, \
    get_user_appointments, get_user_predictions, get_all_appointments, get_all_users, update_user_role, delete_user, \
    update_appointment_status_and_notes
from ai_model import predict_image, CLASS_NAMES_VERBOSE

# --- Session State Initialization ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'user_role' not in st.session_state:  # NEW: Store user role
    st.session_state['user_role'] = None
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'  # Default page
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'prediction_id' not in st.session_state:
    st.session_state['prediction_id'] = None

# Initialize database on app startup
init_db()


# --- Authentication Functions ---
def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(stored_hash, provided_password):
    """Verify a provided password against the stored hash."""
    return stored_hash == hash_password(provided_password)


def login_user(username, password):
    """Authenticate a user."""
    user_record = get_user_by_username(username)
    if user_record:
        user_id, db_username, stored_hash, role = user_record  # Get role from DB
        if verify_password(stored_hash, password):
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            st.session_state['user_id'] = user_id
            st.session_state['user_role'] = role  # Store role
            st.rerun()  # Refresh the page to show main interface
            return True
    return False


def register_new_user(username, password, email=None, role='user'):  # Add role parameter
    """Register a new user."""
    password_hash = hash_password(password)
    return create_user(username, password_hash, email, role)


# --- Role Check Helper Functions ---
def is_admin():
    return st.session_state.get('user_role') == 'admin'


def is_dermatologist():
    return st.session_state.get('user_role') == 'dermatologist'


def is_user():
    return st.session_state.get('user_role') == 'user'


# --- Page Navigation Functions ---
def nav_to_home():
    st.session_state['page'] = 'home'
    st.rerun()


def nav_to_analyze():
    st.session_state['page'] = 'analyze'
    st.rerun()


def nav_to_history():
    st.session_state['page'] = 'history'
    st.rerun()


def nav_to_appointments():
    st.session_state['page'] = 'appointments'
    st.rerun()


def nav_to_view_all_appointments():  # NEW: Navigate to dermatologist/admin view
    st.session_state['page'] = 'view_all_appointments'
    st.rerun()


def nav_to_manage_users():  # NEW: Navigate to admin view
    st.session_state['page'] = 'manage_users'
    st.rerun()


def nav_to_login():
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.session_state['user_id'] = None
    st.session_state['user_role'] = None  # Clear role
    st.session_state['page'] = 'login'
    st.rerun()


# --- Main App Logic ---
def main():
    # Set page config
    st.set_page_config(
        page_title="Skin Care Connect",
        page_icon="🩺",  # Changed icon to be more relevant to skin care
        layout="wide"
    )

    # Sidebar navigation (only if authenticated)
    if st.session_state['authenticated']:
        st.sidebar.title(f"Welcome, {st.session_state['username']} ({st.session_state['user_role']})!")  # Show role
        st.sidebar.button("Home", on_click=nav_to_home)

        # Conditional navigation based on role
        if is_user() or is_dermatologist():  # User and Dermatologist can analyze
            st.sidebar.button("Analyze Image", on_click=nav_to_analyze)
        if is_user():  # Only User can see their history/appointments
            st.sidebar.button("History", on_click=nav_to_history)
            st.sidebar.button("Appointments", on_click=nav_to_appointments)
        if is_dermatologist() or is_admin():  # Dermatologist and Admin can view all appointments
            st.sidebar.button("View All Appointments", on_click=nav_to_view_all_appointments)
        if is_admin():  # Only Admin can manage users
            st.sidebar.button("Manage Users", on_click=nav_to_manage_users)

        st.sidebar.button("Logout", on_click=nav_to_login)

    # Main content based on authentication and page state
    if st.session_state['authenticated']:
        if st.session_state['page'] == 'home':
            home_page()
        elif st.session_state['page'] == 'analyze':
            analyze_page()
        elif st.session_state['page'] == 'history':
            history_page()
        elif st.session_state['page'] == 'appointments':
            appointments_page()
        elif st.session_state['page'] == 'view_all_appointments':  # NEW PAGE
            view_all_appointments_page()
        elif st.session_state['page'] == 'manage_users':  # NEW PAGE
            manage_users_page()
    else:
        login_register_page()


def login_register_page():
    # Updated title to reflect the new name
    st.title("Skin Care Connect - Login/Register")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_user(login_username, login_password):
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_email = st.text_input("Email (Optional)", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
        # Hide role selection for standard user registration
        # reg_role = st.selectbox("Role", ["user"], key="reg_role") # Only 'user' for public registration

        if st.button("Register"):
            if reg_password != reg_confirm_password:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                # Standard registration always creates a 'user'
                if register_new_user(reg_username, reg_password, reg_email, 'user'):
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Username or email already exists.")


def home_page():
    # Updated title to reflect the new name
    st.title("🩺 Skin Care Connect")
    role = st.session_state['user_role']
    st.markdown(f"""
    Welcome, **{st.session_state['username']}** ({role})!

    This system can analyze skin images and provide preliminary classification results based on the HAM10000 dataset.

    **Features Available:**
    *   **Analyze Image:** Upload a skin image for AI-powered classification. (Available for User, Dermatologist)
    *   **History:** View your past analyses. (Available for User)
    *   **Appointments:** View your booked appointments. (Available for User)
    *   **View All Appointments:** See all appointments. (Available for Dermatologist, Admin)
    *   **Manage Users:** Add/edit/delete users and assign roles. (Available for Admin)

    Use the sidebar to navigate based on your role.
    """)


def analyze_page():
    # Updated title to reflect the new name
    st.title("Analyze Skin Image")
    st.markdown(
        f"Upload an image for analysis. Current user: **{st.session_state['username']}** ({st.session_state['user_role']})")
    uploaded_file = st.file_uploader(
        "Choose a skin image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the skin lesion"
    )

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess image
            with st.spinner("Analyzing image..."):
                result = predict_image(image)

                # Save result to database
                user_id = st.session_state['user_id']
                # Generate a unique filename for storage (optional, you might just store the original name)
                unique_filename = str(uuid.uuid4()) + "_" + uploaded_file.name
                prediction_id = save_prediction_result(user_id, unique_filename, result)

                # Store results in session state for potential appointment booking
                st.session_state['prediction_result'] = result
                st.session_state['prediction_id'] = prediction_id
                st.session_state['uploaded_image'] = image  # Store image for display on results page if needed

            # Display results
            display_results(result, prediction_id)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image file.")


def display_results(result, prediction_id):
    """Helper function to display prediction results."""
    if result['error']:
        st.error(f"Error processing image: {result['error']}")
    else:
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        all_predictions = result['all_predictions']
        recommendation = result['recommendation']
        needs_appointment = result['needs_appointment']

        st.subheader("Diagnosis Results")

        # Main prediction
        st.markdown(f"**Predicted Class:** {predicted_class}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        # Recommendation
        st.markdown(f"**Recommendation:** {recommendation}")

        # Risk-based message (Melanoma is high risk)
        if predicted_class == "Melanoma":
            st.error("⚠️ HIGH RISK - MELANOMA DETECTED. Seek immediate medical attention.")
        elif confidence > 80:
            st.success(f"✅ High confidence prediction: {predicted_class}")
        elif confidence > 60:
            st.warning(f"⚠️ Moderate confidence: {predicted_class}. Consult a professional.")
        else:
            st.info(f"ℹ️ Low confidence. Consider professional consultation.")

        # Appointment Recommendation
        if needs_appointment:
            st.warning("The system recommends scheduling an appointment based on this analysis.")
            # Button to book appointment for this specific prediction
            if st.button("Book Appointment for this Analysis"):
                st.session_state['page'] = 'book_appointment'  # Navigate to booking page
                st.session_state['booking_prediction_id'] = prediction_id  # Pass the ID
                st.rerun()
        else:
            st.success("No immediate appointment recommended based on this analysis.")

        # Detailed probabilities
        st.subheader("All Class Probabilities")
        prob_df = pd.DataFrame({
            'Class': list(all_predictions.keys()),
            'Probability (%)': list(all_predictions.values())
        }).sort_values(by='Probability (%)', ascending=False)
        st.dataframe(prob_df, use_container_width=True)

        # Bar chart for top 3 predictions
        sorted_preds = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        top_3_classes, top_3_probs = zip(*sorted_preds[:3])

        st.subheader("Top 3 Predictions")
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(top_3_classes, top_3_probs,
                       color=['red' if cls == 'Melanoma' else 'skyblue' for cls in top_3_classes])
        ax.set_xlabel('Probability (%)')
        ax.set_title('Top 3 Class Predictions')
        ax.set_xlim(0, 100)
        # Add value labels
        for bar, prob in zip(bars, top_3_probs):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%',
                    ha='left', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)


def history_page():
    # Updated title to reflect the new name
    st.title("Analysis History")
    st.markdown(
        f"View your past skin image analyses. Current user: **{st.session_state['username']}** ({st.session_state['user_role']})")

    user_id = st.session_state['user_id']
    predictions = get_user_predictions(user_id)

    if predictions:
        for pred in predictions:
            pid, img_filename, pred_class, conf, rec, needs_appt, timestamp = pred
            with st.expander(f"Analysis on {timestamp} - {pred_class} ({conf:.2f}%)"):
                st.write(f"**Image:** {img_filename}")
                st.write(f"**Prediction:** {pred_class}")
                st.write(f"**Confidence:** {conf:.2f}%")
                st.write(f"**Recommendation:** {rec}")
                st.write(f"**Needs Appointment:** {'Yes' if needs_appt else 'No'}")
                st.write(f"**Analysis ID:** {pid}")
                # Optional: If you stored the image path, you could display it here
                # st.image(f"path/to/stored/images/{img_filename}", caption="Original Image", width=200)
    else:
        st.info("You have no analysis history yet.")


def appointments_page():
    # Updated title to reflect the new name
    st.title("My Appointments")
    st.markdown(
        f"Manage your booked appointments. Current user: **{st.session_state['username']}** ({st.session_state['user_role']})")

    user_id = st.session_state['user_id']
    appointments = get_user_appointments(user_id)

    if appointments:
        for appt in appointments:
            aid, sched_date, status, notes, derm_notes, pred_class, pred_conf, pred_time = appt  # Include derm_notes
            with st.container():
                st.write(f"**Date & Time:** {sched_date}")
                st.write(f"**Status:** {status}")
                st.write(f"**Patient Notes:** {notes or 'N/A'}")
                st.write(f"**Dermatologist Reply:** {derm_notes or 'N/A'}")  # Show dermatologist notes
                st.write(f"**Related Prediction:** {pred_class} ({pred_conf:.2f}%) from {pred_time}")
                st.divider()
    else:
        st.info("You have no upcoming appointments.")


# --- NEW PAGE: View All Appointments (Dermatologist/Admin) ---
def view_all_appointments_page():
    st.title("View All Appointments")
    st.markdown(
        f"See all booked appointments. Current user: **{st.session_state['username']}** ({st.session_state['user_role']})")

    if not (is_dermatologist() or is_admin()):
        st.error("Access denied. Only Dermatologists or Admins can view this page.")
        st.button("Back to Home", on_click=nav_to_home)
        return

    appointments = get_all_appointments()

    if appointments:
        for appt in appointments:
            aid, patient_username, sched_date, status, patient_notes, derm_notes, pred_class, pred_conf, pred_time = appt

            with st.container(border=True):
                st.write(f"**ID:** {aid}")
                st.write(f"**Patient:** {patient_username}")
                st.write(f"**Scheduled Date:** {sched_date}")
                st.write(f"**Status:** {status}")
                st.write(f"**Patient Notes:** {patient_notes or 'N/A'}")
                st.write(f"**Dermatologist Reply:** {derm_notes or 'N/A'}")  # Show dermatologist notes
                st.write(f"**Related Prediction:** {pred_class} ({pred_conf:.2f}%) from {pred_time}")

                # --- NEW: Form for Dermatologist to update status and add notes ---
                if is_dermatologist() or is_admin():  # Allow both roles to update
                    with st.form(key=f"update_form_{aid}"):
                        new_status = st.selectbox("Update Status", ["pending", "confirmed", "cancelled"],
                                                  index=["pending", "confirmed", "cancelled"].index(status),
                                                  key=f"status_{aid}")
                        new_notes = st.text_area("Reply to Patient", value=derm_notes or "", key=f"notes_{aid}")
                        submit_update = st.form_submit_button(label="Update Appointment")

                        if submit_update:
                            success = update_appointment_status_and_notes(aid, new_status, new_notes)
                            if success:
                                st.success(f"Appointment {aid} updated successfully!")
                                st.rerun()  # Refresh the page to show the updated data
                            else:
                                st.error(f"Failed to update appointment {aid}.")
                # --- END NEW FORM ---

                st.divider()
    else:
        st.info("There are no appointments scheduled.")


# --- NEW PAGE: Manage Users (Admin) ---
def manage_users_page():
    st.title("Manage Users")
    st.markdown(
        f"Add, edit, or delete users. Current user: **{st.session_state['username']}** ({st.session_state['user_role']})")

    if not is_admin():
        st.error("Access denied. Only Admins can access this page.")
        st.button("Back to Home", on_click=nav_to_home)
        return

    users = get_all_users()
    if users:
        df = pd.DataFrame(users, columns=['ID', 'Username', 'Email', 'Role', 'Created At'])
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                                   disabled=['ID', 'Username', 'Email', 'Created At'])

        # Handle updates/delete from the editor
        if st.button("Save Changes"):
            changes_made = False
            for index, row in edited_df.iterrows():
                original_row = df.iloc[index]
                if row['Role'] != original_row['Role']:
                    success = update_user_role(row['ID'], row['Role'])
                    if success:
                        st.success(f"Updated role for user ID {row['ID']} to {row['Role']}")
                        changes_made = True
                    else:
                        st.error(f"Failed to update role for user ID {row['ID']}")
                # Check if row was marked for deletion (assuming a 'deleted' column exists in edited_df, which Streamlit data_editor doesn't add by default)
                # For deletion, we need a separate mechanism.
            if changes_made:
                st.rerun()  # Refresh the page to show updated data

        # Deletion mechanism (example using selectbox and button)
        st.subheader("Delete User")
        user_ids_to_delete = st.multiselect("Select User IDs to Delete", options=df['ID'].tolist())
        if st.button("Delete Selected Users") and user_ids_to_delete:
            for uid in user_ids_to_delete:
                success = delete_user(uid)
                if success:
                    st.success(f"Deleted user with ID {uid}")
                else:
                    st.error(f"Failed to delete user with ID {uid}")
            st.rerun()  # Refresh after deletion
    else:
        st.info("There are no users registered.")

    # Add New User Form (Admin only)
    st.subheader("Add New User (Admin)")
    with st.form(key='add_user_form'):
        new_username = st.text_input("New Username")
        new_email = st.text_input("New Email (Optional)")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["user", "dermatologist", "admin"])
        add_submit = st.form_submit_button(label='Add User')

        if add_submit:
            if new_password:  # Ensure password is provided
                if register_new_user(new_username, new_password, new_email, new_role):
                    st.success(f"User '{new_username}' added successfully with role '{new_role}'.")
                    st.rerun()  # Refresh the user list
                else:
                    st.error("Username or email already exists.")
            else:
                st.error("Password is required.")


def book_appointment_page():
    """Page for booking an appointment for a specific prediction."""
    # Updated title to reflect the new name
    st.title("Book Appointment")

    # Get the prediction ID from session state (passed from analyze page)
    pred_id = st.session_state.get('booking_prediction_id')
    if not pred_id:
        st.error("No prediction selected for booking.")
        st.button("Back to Home", on_click=nav_to_home)
        return

    # Fetch prediction details (you might want to add a function in database.py for this)
    # For now, let's assume the details are in session state from the analysis
    result = st.session_state.get('prediction_result')
    if not result:
        st.error("Prediction details not found.")
        st.button("Back to Home", on_click=nav_to_home)
        return

    st.write(f"Booking an appointment for analysis ID: **{pred_id}**")
    st.write(f"Analysis Result: **{result['predicted_class']}** ({result['confidence']:.2f}%)")
    st.write(f"Recommendation: {result['recommendation']}")

    with st.form(key='appointment_form'):
        scheduled_date = st.date_input("Select Date")
        scheduled_time = st.time_input("Select Time")
        notes = st.text_area("Notes (Optional)")

        submit_button = st.form_submit_button(label='Confirm Appointment')

        if submit_button:
            if scheduled_date and scheduled_time:
                try:
                    # Combine date and time into a datetime object
                    scheduled_datetime = datetime.combine(scheduled_date, scheduled_time)
                    user_id = st.session_state['user_id']

                    # Call the booking function
                    book_appointment(user_id, pred_id, scheduled_datetime, notes)

                    st.success("Appointment booked successfully!")
                    # Optionally clear the booking state
                    # del st.session_state['booking_prediction_id']
                    # del st.session_state['prediction_result']
                    st.button("Back to Appointments", on_click=nav_to_appointments)
                except Exception as e:
                    st.error(f"Error booking appointment: {str(e)}")
            else:
                st.error("Please select both a date and a time.")

    st.button("Cancel", on_click=lambda: nav_to_analyze())  # Go back to analyze page


if __name__ == "__main__":
    # Check if we are on the booking page and render it separately
    if st.session_state.get('page') == 'book_appointment':
        book_appointment_page()
    else:
        main()