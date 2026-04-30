import io
import os
import sys
import time
import base64
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb

# Ensure project root is importable.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.preprocessing import preprocess_df, split_clients
from backend.encryption import create_ckks_context, encrypt_matrix, decrypt_matrix
from backend.blockchain import BlockchainClient
from backend.client1 import train_client_1
from backend.client2 import train_client_2
from backend.global_model import evaluate_global_model


st.set_page_config(page_title="HealthTrust-FL", layout="wide")


def apply_theme() -> None:
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

    theme = st.session_state.theme
    if theme == "dark":
        bg = "#0b1220"
        card = "#111827"
        text = "#f9fafb"
        muted = "#9ca3af"
        border = "#334155"
        accent = "#0284c7"
    else:
        bg = "#ffffff"
        card = "#f3f4f6"
        text = "#111827"
        muted = "#4b5563"
        border = "#d1d5db"
        accent = "#0369a1"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --card: {card};
            --text: {text};
            --muted: {muted};
            --border: {border};
            --accent: {accent};
        }}
        .stApp {{
            background: var(--bg);
            color: var(--text);
        }}
        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--card), var(--bg));
            border-right: 1px solid var(--border);
        }}
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: var(--text) !important;
        }}
        .hero {{
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--card);
        }}
        .nav-btn .stButton > button {{
            width: 100%;
            text-align: left;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text);
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}
        .nav-btn.active .stButton > button {{
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }}
        .muted {{
            color: var(--muted);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    defaults = {
        "page": "Home",
        "uploaded_name": None,
        "dataset_df": None,
        "X": None,
        "y": None,
        "results": None,
        "pipeline_details": None,
        "last_trained": None,
        "deployed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> str:
    _init_state()
    st.sidebar.title("HealthTrust-FL")
    st.sidebar.caption("Secure and Compliant Federated Learning")

    nav_items = [
        "Home",
        "Data Pipeline",
        "Training",
        "Results",
        "Predictions",
        "Deploy",
        "Exit",
    ]

    for item in nav_items:
        active = st.session_state.page == item
        css = "nav-btn active" if active else "nav-btn"
        with st.sidebar.container():
            st.markdown(f"<div class='{css}'>", unsafe_allow_html=True)
            if st.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state.page = item
            st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.divider()
    is_dark = st.sidebar.toggle("Dark Mode", value=st.session_state.theme == "dark")
    st.session_state.theme = "dark" if is_dark else "light"
    return st.session_state.page


def _parse_chunks(entries):
    completed = []
    buffers = {}
    totals = {}
    for entry in entries:
        try:
            model_id, part_info, chunk = entry.split(":", 2)
            part_str, total_str = part_info.split("/", 1)
            part = int(part_str)
            total = int(total_str)
        except ValueError:
            continue

        if model_id not in buffers:
            buffers[model_id] = {}
        buffers[model_id][part] = chunk
        totals[model_id] = total

        if len(buffers[model_id]) == total:
            reassembled = "".join(buffers[model_id][i] for i in range(1, total + 1))
            completed.append((model_id, reassembled))
            buffers.pop(model_id, None)
            totals.pop(model_id, None)
    return completed


def _ensemble_predict_proba(x_data: np.ndarray, bc: BlockchainClient):
    weights = bc.get_weights()
    models = _parse_chunks(weights)
    if not models:
        return None

    dmat = xgb.DMatrix(x_data)
    probs = np.zeros(len(x_data))
    for _, model_b64 in models:
        booster = xgb.Booster()
        booster.load_model(bytearray(base64.b64decode(model_b64.encode("utf-8"))))
        probs += booster.predict(dmat)
    probs /= len(models)
    return probs


def run_pipeline(file_bytes: bytes) -> None:
    provider_url = os.environ.get("FL_PROVIDER_URL", "http://127.0.0.1:7545")
    contract_json = os.environ.get(
        "FL_CONTRACT_JSON",
        str(ROOT_DIR / "blockchain" / "build" / "contracts" / "FLContract.json"),
    )
    contract_address = os.environ.get(
        "FL_CONTRACT_ADDRESS", "0x6Bb9e2b463bcb5c1F4FED7983dB5f98515DE52d9"
    )

    with st.spinner("Running secure federated pipeline..."):
        progress = st.progress(0)
        step_times = {}

        t0 = time.perf_counter()
        raw_df = pd.read_csv(io.BytesIO(file_bytes))
        raw_df = raw_df.head(800).copy()
        x_data, y_data, _ = preprocess_df(raw_df)
        step_times["preprocessing"] = time.perf_counter() - t0
        progress.progress(15)

        t1 = time.perf_counter()
        ctx = create_ckks_context()
        enc_rows = encrypt_matrix(ctx, x_data)
        x_dec = decrypt_matrix(enc_rows)
        step_times["encryption"] = time.perf_counter() - t1
        progress.progress(30)

        t2 = time.perf_counter()
        (x1, y1), (x2, y2) = split_clients(x_dec, y_data)
        step_times["split"] = time.perf_counter() - t2
        progress.progress(45)

        bc = BlockchainClient(provider_url, contract_json, contract_address)
        c1 = train_client_1(x1, y1, bc)
        progress.progress(60)
        c2 = train_client_2(x2, y2, bc)
        progress.progress(80)
        g = evaluate_global_model(x_dec, y_data, bc)
        progress.progress(100)

    st.session_state.dataset_df = raw_df
    st.session_state.X = x_dec
    st.session_state.y = y_data
    st.session_state.results = {"client1": c1, "client2": c2, "global": g}
    st.session_state.pipeline_details = {
        "step_times": step_times,
        "client_sizes": {"client1": len(x1), "client2": len(x2)},
        "blockchain_stored": bool(c1.get("tx_hash")) and bool(c2.get("tx_hash")),
        "weights_generated": 2,
        "total_rows_used": len(raw_df),
        "total_time": sum(step_times.values())
        + c1["time"]
        + c2["time"]
        + g["global_time"],
    }
    st.session_state.last_trained = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_home() -> None:
    st.title("HealthTrust-FL")
    st.caption("Secure and Compliant Federated Learning")

    with st.container(border=True):
        st.markdown(
            "Complete federated learning workflow dashboard for Chronic Kidney Disease analysis "
            "with encryption and blockchain-backed model updates."
        )
        st.markdown(
            "`Preprocessing -> Encryption -> Client Training -> Blockchain -> Global Model -> Prediction`"
        )


def render_pipeline() -> None:
    st.header("Data Pipeline")
    uploaded = st.file_uploader(
        "Upload CKD EHR CSV", type=["csv"], key="uploader_pipeline"
    )
    if uploaded is not None:
        st.session_state.uploaded_name = uploaded.name
        st.write(f"File: `{uploaded.name}`")
        if st.button("Run Federated Learning", type="primary"):
            try:
                run_pipeline(uploaded.getvalue())
                st.success("Pipeline execution completed.")
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")

    details = st.session_state.pipeline_details
    if not details:
        st.info("Run the pipeline to view step-wise workflow details.")
        return

    step_times = details["step_times"]
    sizes = details["client_sizes"]

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("1. Data Preprocessing")
            st.write("Status: Completed")
            st.write("Includes: Missing value handling, encoding, normalization")
            st.write(f"Time taken: {step_times['preprocessing']:.3f} sec")
    with c2:
        with st.container(border=True):
            st.subheader("2. Dataset Split")
            st.write("Status: Completed")
            st.write(f"Client 1 size: {sizes['client1']}")
            st.write(f"Client 2 size: {sizes['client2']}")
            st.write(f"Time taken: {step_times['split']:.3f} sec")
    with c3:
        with st.container(border=True):
            st.subheader("3. Homomorphic Encryption")
            st.write("Status: Completed")
            st.write("CKKS encryption and decryption applied")
            st.write(f"Time taken: {step_times['encryption']:.3f} sec")


def render_training() -> None:
    st.header("Training")
    results = st.session_state.results
    details = st.session_state.pipeline_details
    if not results or not details:
        st.info("Run the pipeline from Data Pipeline to view training details.")
        return

    c1 = results["client1"]
    c2 = results["client2"]
    g = results["global"]

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Client 1 Training")
            st.write(f"Accuracy: {c1['accuracy']:.4f}")
            st.write(f"Time taken: {c1['time']:.3f} sec")
            st.write("Weights generated: Yes")
            st.write(f"Transaction: {c1.get('tx_hash', 'N/A')}")
    with col2:
        with st.container(border=True):
            st.subheader("Client 2 Training")
            st.write(f"Accuracy: {c2['accuracy']:.4f}")
            st.write(f"Time taken: {c2['time']:.3f} sec")
            st.write("Weights generated: Yes")
            st.write(f"Transaction: {c2.get('tx_hash', 'N/A')}")

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.subheader("Blockchain Storage")
            status = "Confirmed" if details["blockchain_stored"] else "Not confirmed"
            st.write(f"Status: {status}")
            st.write(f"Total weight updates: {details['weights_generated']}")
    with col4:
        with st.container(border=True):
            st.subheader("Global Model Aggregation")
            st.write(f"Global Accuracy: {g['global_accuracy']:.4f}")
            st.write(f"Global Time: {g['global_time']:.3f} sec")
            st.write(f"Total Pipeline Time: {details['total_time']:.3f} sec")


def _draw_bar(ax, labels, values, colors, title, ylabel, legend_name):
    bars = ax.bar(labels, values, color=colors, label=legend_name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom"
        )


def render_summary() -> None:
    results = st.session_state.results
    details = st.session_state.pipeline_details
    if not results or not details:
        st.info("Run the pipeline to generate summary report.")
        return

    st.subheader("Final Summary Report")
    c1 = results["client1"]
    c2 = results["client2"]
    g = results["global"]
    step_times = details["step_times"]

    c_left, c_right = st.columns(2)
    with c_left:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        _draw_bar(
            ax1,
            ["Client1", "Client2", "Global"],
            [c1["accuracy"], c2["accuracy"], g["global_accuracy"]],
            ["#2563eb", "#16a34a", "#f59e0b"],
            "Accuracy Summary",
            "Accuracy",
            "Accuracy",
        )
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

    with c_right:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        _draw_bar(
            ax2,
            ["Client1", "Client2", "Global"],
            [c1["time"], c2["time"], g["global_time"]],
            ["#7c3aed", "#ef4444", "#06b6d4"],
            "Computation Time Summary",
            "Seconds",
            "Seconds",
        )
        st.pyplot(fig2)

    if st.session_state.X is not None:
        probs = _predict_probabilities_safe(st.session_state.X)
        if probs is not None and len(probs) > 0:
            idx = min(0, len(probs) - 1)
            pred = (
                "Chronic Kidney Disease (CKD)"
                if probs[idx] >= 0.5
                else "Non Chronic Kidney Disease (NCKD)"
            )
            with st.container(border=True):
                st.write(f"Sample Person 0 Prediction: {pred}")

            max_people = min(20, len(probs))
            group_preds = (probs[:max_people] >= 0.5).astype(int)
            fig3, ax3 = plt.subplots(figsize=(8, 3.5))
            bars = ax3.bar(
                [str(i) for i in range(max_people)],
                group_preds,
                color=["#ef4444" if v == 1 else "#10b981" for v in group_preds],
                label="Prediction",
            )
            ax3.set_title("Group Prediction Summary")
            ax3.set_xlabel("Person")
            ax3.set_ylabel("Prediction (1=CKD, 0=NCKD)")
            ax3.legend()
            for bar in bars:
                h = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.02,
                    f"{int(h)}",
                    ha="center",
                )
            st.pyplot(fig3)


def render_results() -> None:
    st.header("Results")
    results = st.session_state.results
    details = st.session_state.pipeline_details
    if not results or not details:
        st.info("Run the pipeline from Data Pipeline to view results.")
        return

    c1 = results["client1"]
    c2 = results["client2"]
    g = results["global"]
    step_times = details["step_times"]

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric("Client 1 Accuracy", f"{c1['accuracy']:.4f}")
    with col2:
        with st.container(border=True):
            st.metric("Client 2 Accuracy", f"{c2['accuracy']:.4f}")
    with col3:
        with st.container(border=True):
            st.metric("Global Accuracy", f"{g['global_accuracy']:.4f}")

    c_left, c_right = st.columns(2)
    with c_left:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        _draw_bar(
            ax1,
            ["Client1", "Client2", "Global"],
            [c1["accuracy"], c2["accuracy"], g["global_accuracy"]],
            ["#2563eb", "#16a34a", "#f59e0b"],
            "Accuracy Comparison",
            "Accuracy",
            "Accuracy",
        )
        ax1.set_ylim(0, 1)
        st.pyplot(fig1)

    with c_right:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        _draw_bar(
            ax2,
            ["Client1", "Client2", "Global"],
            [c1["time"], c2["time"], g["global_time"]],
            ["#7c3aed", "#ef4444", "#06b6d4"],
            "Computation Time Comparison",
            "Seconds",
            "Seconds",
        )
        st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 3.3))
    _draw_bar(
        ax3,
        ["Preprocessing", "Encryption", "Split"],
        [step_times["preprocessing"], step_times["encryption"], step_times["split"]],
        ["#0ea5e9", "#84cc16", "#f97316"],
        "Pipeline Step Time Comparison",
        "Seconds",
        "Step Time",
    )
    st.pyplot(fig3)

    render_summary()


def _draw_gauge(value: float, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.set_aspect("equal")
    ax.axis("off")
    angle = np.interp(value, [0, 1], [-np.pi / 2, np.pi / 2])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, linewidth=10, color="#d1d5db"))
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=6, color="#0ea5e9")
    ax.text(0, -0.1, f"{value:.2%}", ha="center", fontsize=14, fontweight="bold")
    ax.text(0, -0.35, title, ha="center", fontsize=10)
    return fig


def _predict_probabilities_safe(x_data: np.ndarray):
    try:
        provider_url = os.environ.get("FL_PROVIDER_URL", "http://127.0.0.1:7545")
        contract_json = os.environ.get(
            "FL_CONTRACT_JSON",
            str(ROOT_DIR / "blockchain" / "build" / "contracts" / "FLContract.json"),
        )
        contract_address = os.environ.get(
            "FL_CONTRACT_ADDRESS", "0x9345AC6FCA4552a2CDaa1A905Ff3E3A8F4d647B0"
        )
        bc = BlockchainClient(provider_url, contract_json, contract_address)
        return _ensemble_predict_proba(x_data, bc)
    except Exception:
        return None


def render_predictions() -> None:
    st.header("Predictions")
    x_data = st.session_state.X
    y_data = st.session_state.y
    if x_data is None or y_data is None:
        st.info("Run the pipeline from Data Pipeline to enable predictions.")
        return

    probs = _predict_probabilities_safe(x_data)
    if probs is None:
        st.warning(
            "Prediction models are unavailable from blockchain. Run pipeline again."
        )
        return

    n = len(x_data)
    with st.container(border=True):
        st.subheader("Individual Prediction")
        idx = st.number_input(
            "Person index", min_value=0, max_value=n - 1, value=0, step=1
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Predict Selected Person"):
                st.session_state.pred_idx_1 = int(idx)
            if st.button("Select Random Person"):
                st.session_state.pred_idx_1 = int(np.random.randint(0, n))
        with col_b:
            idx1 = st.session_state.get("pred_idx_1", int(idx))
            p1 = probs[idx1]
            label1 = (
                "Chronic Kidney Disease (CKD)"
                if p1 >= 0.5
                else "Non Chronic Kidney Disease (NCKD)"
            )
            st.write(f"Selected index: {idx1}")
            st.write(f"Prediction Result: {label1}")
            st.pyplot(_draw_gauge(float(p1), "Prediction Confidence"))

            contrib = np.abs(x_data[idx1])
            top_k = min(10, len(contrib))
            top_idx = np.argsort(contrib)[-top_k:]
            fig_c1, ax_c1 = plt.subplots(figsize=(6, 3))
            ax_c1.bar(
                [str(i) for i in top_idx],
                contrib[top_idx],
                color="#2563eb",
                label="Contribution",
            )
            ax_c1.set_title("Feature Contribution")
            ax_c1.set_xlabel("Feature Index")
            ax_c1.set_ylabel("Magnitude")
            ax_c1.legend()
            st.pyplot(fig_c1)

    with st.container(border=True):
        st.subheader("Second Person Prediction")
        idx2 = st.number_input(
            "Second person index",
            min_value=0,
            max_value=n - 1,
            value=min(1, n - 1),
            step=1,
        )
        if st.button("Predict Second Person"):
            st.session_state.pred_idx_2 = int(idx2)
        selected2 = st.session_state.get("pred_idx_2", int(idx2))
        p2 = probs[selected2]
        label2 = (
            "Chronic Kidney Disease (CKD)"
            if p2 >= 0.5
            else "Non Chronic Kidney Disease (NCKD)"
        )
        st.write(f"Selected index: {selected2}")
        st.write(f"Prediction Result: {label2}")
        st.pyplot(_draw_gauge(float(p2), "Prediction Confidence"))

        contrib2 = np.abs(x_data[selected2])
        top_k2 = min(10, len(contrib2))
        top_idx2 = np.argsort(contrib2)[-top_k2:]
        fig_c2, ax_c2 = plt.subplots(figsize=(6, 3))
        ax_c2.bar(
            [str(i) for i in top_idx2],
            contrib2[top_idx2],
            color="#10b981",
            label="Contribution",
        )
        ax_c2.set_title("Feature Contribution (Second Person)")
        ax_c2.set_xlabel("Feature Index")
        ax_c2.set_ylabel("Magnitude")
        ax_c2.legend()
        st.pyplot(fig_c2)

    with st.container(border=True):
        st.subheader("Group Prediction")
        group_n = st.slider(
            "Number of persons", min_value=2, max_value=min(50, n), value=min(10, n)
        )
        if st.button("Run Group Prediction"):
            preds = (probs[:group_n] >= 0.5).astype(int)
            fig_g, ax_g = plt.subplots(figsize=(8, 3.2))
            bars = ax_g.bar(
                [str(i) for i in range(group_n)],
                preds,
                color=["#ef4444" if v == 1 else "#10b981" for v in preds],
                label="Prediction",
            )
            ax_g.set_title("Person vs Prediction")
            ax_g.set_xlabel("Person")
            ax_g.set_ylabel("Prediction (1=CKD, 0=NCKD)")
            ax_g.legend()
            for bar in bars:
                h = bar.get_height()
                ax_g.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.02,
                    f"{int(h)}",
                    ha="center",
                )
            st.pyplot(fig_g)


def render_deploy() -> None:
    st.header("Deploy")
    with st.container(border=True):
        st.write("Model Status: Ready")
        st.write(f"Last Trained: {st.session_state.last_trained or 'Not trained yet'}")
        if st.button("Deploy Model"):
            st.session_state.deployed = True
            st.success("Model deployed successfully.")


def render_exit() -> None:
    st.header("Exit")
    with st.container(border=True):
        if st.button("Exit Application", type="primary"):
            st.warning("Application Closed")
            st.stop()


def main() -> None:
    apply_theme()
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Data Pipeline":
        render_pipeline()
    elif page == "Training":
        render_training()
    elif page == "Results":
        render_results()
    elif page == "Predictions":
        render_predictions()
    elif page == "Deploy":
        render_deploy()
    elif page == "Exit":
        render_exit()


if __name__ == "__main__":
    main()
