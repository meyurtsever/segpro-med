# 🧠 Crowdsourcing Feature Specification for Medical Annotation Tool

---

## 📁 1. Project Overview

This document outlines the step-by-step integration of a **Crowdsourcing Workflow** into an existing Gradio-based medical data annotation application. The system introduces a role-based login system and extends the application to allow **admins** to manage datasets and assign tasks to **experts**, while experts contribute annotations using the existing Editor tab.

---

## ⚙️ 2. System-Level Switch

### 🔘 Authentication Toggle
- Add a boolean flag to `app.py`:  
  ```python
  ENABLE_AUTH = True
  ```
- If `ENABLE_AUTH` is `False`:  
  - The app launches **without login** and behaves as usual.
- If `ENABLE_AUTH` is `True`:  
  - A login screen appears before any other part of the app loads.

---

## 🔐 3. Login System

### 🗂️ `db/users.txt` Format
Store users in a simple text format:
```
id,password,role,expert_score
john_doe,pass123,expert,4.2
admin1,adminpass,admin,5.0
```

### 🔐 Gradio Login Page
- Create a new page (`gr.Blocks`) for login.
- Fields:
  - `Username` (Text)
  - `Password` (Password field)
- Validate credentials from `db/users.txt`.
- If login successful:
  - Store `user_id`, `role`, and `expert_score` in a session/state variable.
  - Redirect user to the main app interface.

---

## 🧑‍💼 4. Role-Based Tabs After Login

Depending on role from login:

| Role   | Additional Tab     |
|--------|--------------------|
| admin  | `Management`       |
| expert | `Contribute`       |

These tabs appear **in addition to** the normal tabs.

---

## 📊 5. Admin: Management Tab

### 🔧 5.1 Create Crowdsourcing Campaign

- Admin can start a new crowdsourcing event.
- Admin provides the **dataset path** via:
  - Directory browser
  - Textbox

#### 🔍 5.2 Dataset Scan Logic

- Upon selecting a folder, recursively count **subfolders**.
- Each subfolder represents **a patient** if it contains known modalities:
  - `FLAIR`, `T1`, `T1c`, `T2` (case insensitive)
- Display:
  - Total number of patients
  - Total number of modality folders (optional)

#### 🧾 5.3 Create Crowdsourcing Card

Each crowdsourcing campaign is shown as a **card**:
- Title: Campaign name or root folder name
- Info:
  - Total Patients
  - Assigned Patients
  - Annotated
  - Reviewed
- Progress bar to show completion status
- “Details” button to open assignment panel

---

### 🧩 5.4 Campaign Details

- In card details:
  - Assign tasks to experts
  - View all assigned folders and to whom
  - Track progress per expert
- Display dropdown of available `expert` users
- Show unassigned patients from this campaign
- Admin selects patient folders and assigns them to a selected expert

---

## 🧑‍🔬 6. Expert: Contribute Tab

### 📝 6.1 Assigned Patients UI

- Display a dropdown:
  - Lists patient IDs or folder names assigned to the logged-in expert.
- Selecting an entry triggers loading of the related dataset.

### 🧠 6.2 Reuse `Editor` Tab for Annotation

- Load the dataset into the `Editor` tab for annotation.
- Pass metadata to the Editor tab indicating that:
  - The load was initiated **from crowdsourcing**
  - A `crowdsourcing_mode=True` flag is active

### 🔒 6.3 Conditional UI for Crowdsourcing

In the Editor tab:
- If `crowdsourcing_mode=True`:
  - Display **extra modules/buttons**:
    - `Submit Annotation`
    - `Send for Review`
    - Possibly `View Instructions`
- These modules are hidden when accessed normally.

---

## 🧬 7. Metadata and State Handling

### 💾 Track Task State

Maintain task state (can be a `db/assignments.txt` or JSON):

```json
{
  "campaign_001": {
    "assigned": {
      "john_doe": ["patient_01", "patient_02"]
    },
    "completed": {
      "john_doe": ["patient_01"]
    }
  }
}
```

---

## 📊 8. Progress and Status

Admin-side cards and campaign pages must show:
- ✅ Total patients
- 👨‍⚕️ Assigned to experts
- 🖍️ Annotated
- 📤 Submitted for review

Use progress bars or % indicators in the UI.

---

## 🧪 9. Testing & Demonstration Notes

For MVP/testing/demo purposes:
- Use `.txt` or `.json` files instead of a database
- Maintain logs for:
  - Login attempts
  - Dataset creation
  - Assignment changes
  - Annotation submissions

---
