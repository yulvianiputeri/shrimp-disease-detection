import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def render_evaluasi(knn, svm, voting, X_test_eval, y_test_eval):
    """Render Evaluasi Model page"""
    
    st.markdown("""<div class="section-header">Evaluasi Model Klasifikasi</div>""", unsafe_allow_html=True)
    
    try:
        # Make predictions
        knn_pred_eval = knn.predict(X_test_eval)
        svm_pred_eval = svm.predict(X_test_eval)
        voting_pred_eval = voting.predict(X_test_eval)
        
        # Calculate accuracy
        knn_acc = (knn_pred_eval == y_test_eval).mean() * 100
        svm_acc = (svm_pred_eval == y_test_eval).mean() * 100
        voting_acc = (voting_pred_eval == y_test_eval).mean() * 100
        
        # Display metrics - 3 KOLOM
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size: 36px;">{knn_acc:.2f}%</div>
            <div class="metric-label">Akurasi KNN</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size: 36px;">{svm_acc:.2f}%</div>
            <div class="metric-label">Akurasi SVM</div>
            </div>""", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""<div class="metric-card" style="border: 3px solid #667eea;">
            <div class="metric-value" style="font-size: 36px; color: #667eea;">{voting_acc:.2f}%</div>
            <div class="metric-label" style="font-weight: 600;">Akurasi Voting Ensemble</div>
            </div>""", unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("""<div class="evaluation-card">
        <div class="eval-title">Confusion Matrix</div>""", unsafe_allow_html=True)
        
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        
        with col_cm1:
            st.markdown("<div style='text-align: center; font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>KNN</div>", unsafe_allow_html=True)
            cm_knn = confusion_matrix(y_test_eval, knn_pred_eval)
            fig_knn, ax_knn = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax_knn,
                       xticklabels=['Sehat', 'Sakit'], yticklabels=['Sehat', 'Sakit'])
            ax_knn.set_xlabel('Prediksi')
            ax_knn.set_ylabel('Aktual')
            st.pyplot(fig_knn)
            plt.close()
        
        with col_cm2:
            st.markdown("<div style='text-align: center; font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>SVM</div>", unsafe_allow_html=True)
            cm_svm = confusion_matrix(y_test_eval, svm_pred_eval)
            fig_svm, ax_svm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=ax_svm,
                       xticklabels=['Sehat', 'Sakit'], yticklabels=['Sehat', 'Sakit'])
            ax_svm.set_xlabel('Prediksi')
            ax_svm.set_ylabel('Aktual')
            st.pyplot(fig_svm)
            plt.close()
        
        with col_cm3:
            st.markdown("<div style='text-align: center; font-weight: 600; color: #667eea; margin-bottom: 10px;'>Voting Ensemble</div>", unsafe_allow_html=True)
            cm_voting = confusion_matrix(y_test_eval, voting_pred_eval)
            fig_voting, ax_voting = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Purples', ax=ax_voting,
                       xticklabels=['Sehat', 'Sakit'], yticklabels=['Sehat', 'Sakit'])
            ax_voting.set_xlabel('Prediksi')
            ax_voting.set_ylabel('Aktual')
            st.pyplot(fig_voting)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Classification Report
        st.markdown("""<div class="evaluation-card">
        <div class="eval-title">Classification Report</div>""", unsafe_allow_html=True)
        
        col_rpt1, col_rpt2, col_rpt3 = st.columns(3)
        
        with col_rpt1:
            st.markdown("<div style='font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>KNN</div>", unsafe_allow_html=True)
            report_knn = classification_report(y_test_eval, knn_pred_eval, 
                                               target_names=['Sehat', 'Sakit'], output_dict=True)
            df_rpt_knn = pd.DataFrame(report_knn).transpose()
            st.dataframe(df_rpt_knn.style.format('{:.4f}'), use_container_width=True)
        
        with col_rpt2:
            st.markdown("<div style='font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>SVM</div>", unsafe_allow_html=True)
            report_svm = classification_report(y_test_eval, svm_pred_eval, 
                                               target_names=['Sehat', 'Sakit'], output_dict=True)
            df_rpt_svm = pd.DataFrame(report_svm).transpose()
            st.dataframe(df_rpt_svm.style.format('{:.4f}'), use_container_width=True)
        
        with col_rpt3:
            st.markdown("<div style='font-weight: 600; color: #667eea; margin-bottom: 10px;'>Voting Ensemble</div>", unsafe_allow_html=True)
            report_voting = classification_report(y_test_eval, voting_pred_eval, 
                                                  target_names=['Sehat', 'Sakit'], output_dict=True)
            df_rpt_voting = pd.DataFrame(report_voting).transpose()
            st.dataframe(df_rpt_voting.style.format('{:.4f}'), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ROC Curve
        st.markdown("""<div class="evaluation-card">
        <div class="eval-title">ROC Curve</div>""", unsafe_allow_html=True)
        
        col_roc1, col_roc2, col_roc3 = st.columns(3)
        
        # KNN ROC
        with col_roc1:
            st.markdown("<div style='font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>KNN</div>", unsafe_allow_html=True)
            knn_proba_eval = knn.predict_proba(X_test_eval)[:, 1]
            fpr_knn, tpr_knn, _ = roc_curve(y_test_eval, knn_proba_eval)
            roc_auc_knn = auc(fpr_knn, tpr_knn)
            fig_roc_knn, ax_roc_knn = plt.subplots(figsize=(5, 4))
            ax_roc_knn.plot(fpr_knn, tpr_knn, color='#1B4D3E', lw=2, label=f'AUC = {roc_auc_knn:.4f}')
            ax_roc_knn.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc_knn.set_xlim([0.0, 1.0])
            ax_roc_knn.set_ylim([0.0, 1.05])
            ax_roc_knn.set_xlabel('FPR', fontsize=9)
            ax_roc_knn.set_ylabel('TPR', fontsize=9)
            ax_roc_knn.set_title('ROC - KNN', fontsize=10)
            ax_roc_knn.legend(loc="lower right", fontsize=8)
            st.pyplot(fig_roc_knn)
            plt.close()
        
        # SVM ROC
        with col_roc2:
            st.markdown("<div style='font-weight: 600; color: #1B4D3E; margin-bottom: 10px;'>SVM</div>", unsafe_allow_html=True)
            svm_proba_eval = svm.predict_proba(X_test_eval)[:, 1]
            fpr_svm, tpr_svm, _ = roc_curve(y_test_eval, svm_proba_eval)
            roc_auc_svm = auc(fpr_svm, tpr_svm)
            fig_roc_svm, ax_roc_svm = plt.subplots(figsize=(5, 4))
            ax_roc_svm.plot(fpr_svm, tpr_svm, color='#34CE57', lw=2, label=f'AUC = {roc_auc_svm:.4f}')
            ax_roc_svm.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc_svm.set_xlim([0.0, 1.0])
            ax_roc_svm.set_ylim([0.0, 1.05])
            ax_roc_svm.set_xlabel('FPR', fontsize=9)
            ax_roc_svm.set_ylabel('TPR', fontsize=9)
            ax_roc_svm.set_title('ROC - SVM', fontsize=10)
            ax_roc_svm.legend(loc="lower right", fontsize=8)
            st.pyplot(fig_roc_svm)
            plt.close()
        
        # VOTING ROC
        with col_roc3:
            st.markdown("<div style='font-weight: 600; color: #667eea; margin-bottom: 10px;'>Voting Ensemble</div>", unsafe_allow_html=True)
            voting_proba_eval = voting.predict_proba(X_test_eval)[:, 1]
            fpr_voting, tpr_voting, _ = roc_curve(y_test_eval, voting_proba_eval)
            roc_auc_voting = auc(fpr_voting, tpr_voting)
            fig_roc_voting, ax_roc_voting = plt.subplots(figsize=(5, 4))
            ax_roc_voting.plot(fpr_voting, tpr_voting, color='#667eea', lw=2, label=f'AUC = {roc_auc_voting:.4f}')
            ax_roc_voting.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            ax_roc_voting.set_xlim([0.0, 1.0])
            ax_roc_voting.set_ylim([0.0, 1.05])
            ax_roc_voting.set_xlabel('FPR', fontsize=9)
            ax_roc_voting.set_ylabel('TPR', fontsize=9)
            ax_roc_voting.set_title('ROC - Voting', fontsize=10)
            ax_roc_voting.legend(loc="lower right", fontsize=8)
            st.pyplot(fig_roc_voting)
            plt.close()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Comparison Summary
        st.markdown("""<div class="evaluation-card">
        <div class="eval-title">Perbandingan Model</div>""", unsafe_allow_html=True)
        
        comparison_data = {
            'Metrik': ['Akurasi', 'Precision (Sehat)', 'Precision (Sakit)', 
                      'Recall (Sehat)', 'Recall (Sakit)', 'F1-Score (Sehat)', 
                      'F1-Score (Sakit)', 'AUC-ROC'],
            'KNN': [f"{knn_acc:.2f}%", 
                   f"{report_knn['Sehat']['precision']:.4f}",
                   f"{report_knn['Sakit']['precision']:.4f}",
                   f"{report_knn['Sehat']['recall']:.4f}",
                   f"{report_knn['Sakit']['recall']:.4f}",
                   f"{report_knn['Sehat']['f1-score']:.4f}",
                   f"{report_knn['Sakit']['f1-score']:.4f}",
                   f"{roc_auc_knn:.4f}"],
            'SVM': [f"{svm_acc:.2f}%", 
                   f"{report_svm['Sehat']['precision']:.4f}",
                   f"{report_svm['Sakit']['precision']:.4f}",
                   f"{report_svm['Sehat']['recall']:.4f}",
                   f"{report_svm['Sakit']['recall']:.4f}",
                   f"{report_svm['Sehat']['f1-score']:.4f}",
                   f"{report_svm['Sakit']['f1-score']:.4f}",
                   f"{roc_auc_svm:.4f}"],
            'Voting Ensemble': [f"{voting_acc:.2f}%",
                               f"{report_voting['Sehat']['precision']:.4f}",
                               f"{report_voting['Sakit']['precision']:.4f}",
                               f"{report_voting['Sehat']['recall']:.4f}",
                               f"{report_voting['Sakit']['recall']:.4f}",
                               f"{report_voting['Sehat']['f1-score']:.4f}",
                               f"{report_voting['Sakit']['f1-score']:.4f}",
                               f"{roc_auc_voting:.4f}"]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Highlight best values
        def highlight_best(s):
            if s.name == 'Metrik':
                return [''] * len(s)
            values = []
            for val in s:
                try:
                    values.append(float(val.replace('%', '')))
                except:
                    values.append(0)
            max_val = max(values)
            return ['background-color: #d4edda; font-weight: bold' if v == max_val else '' for v in values]
        
        styled_df = df_comparison.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        
# ============================================================
        # BAR CHART COMPARISON
        # ============================================================
        st.markdown("---")
        st.markdown("### Visualisasi Perbandingan Akurasi")
        
        models = ['KNN', 'SVM', 'Voting Ensemble']
        accuracies = [knn_acc, svm_acc, voting_acc]
        colors = ['#1B4D3E', '#34CE57', '#667eea']
        
        fig_perf, ax_perf = plt.subplots(figsize=(10, 6))
        bars = ax_perf.bar(models, accuracies, color=colors, edgecolor='white', linewidth=2)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax_perf.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax_perf.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax_perf.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
        ax_perf.set_ylim(0, 105)
        ax_perf.grid(axis='y', alpha=0.3, linestyle='--')
        
        st.pyplot(fig_perf)
        plt.close()
        
        # Summary box
        st.success(f"""
        ✅ **Hasil Akhir:**
        - **Voting Ensemble** mencapai accuracy **{voting_acc:.2f}%**
        - Improvement **+{voting_acc - knn_acc:.2f}%** dibanding KNN
        - Improvement **+{voting_acc - svm_acc:.2f}%** dibanding SVM
        - AUC-ROC: **{roc_auc_voting:.4f}** (hampir perfect)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)        
        
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        st.info("Pastikan file X_test.pkl dan y_test.pkl tersedia untuk evaluasi model.")