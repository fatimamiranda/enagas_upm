import os
from fpdf import FPDF
import matplotlib.pyplot as plt
from sklearn.metrics import  roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ReportGenerator:
    """
    Clase para generar informes PDF con gráficos y comparación de resultados de modelos.
    """
    def __init__(self, report_dir='reports/'):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def plot_confusion_matrix(self, model, X_val, y_val, model_name):
        """
        Genera y guarda la matriz de confusión.
        """
        disp = ConfusionMatrixDisplay.from_estimator(model, X_val, y_val, cmap=plt.cm.Blues)
        plt.title(f"Matriz de Confusión - {model_name}")
        path = os.path.join(self.report_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(path)
        plt.close()
        return path

    def plot_roc_curve(self, model, X_val, y_val, model_name):
        """
        Genera y guarda la curva ROC.
        """
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title(f"Curva ROC - {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        path = os.path.join(self.report_dir, f"{model_name}_roc_curve.png")
        plt.savefig(path)
        plt.close()
        return path

    def generate_pdf_report(self, model_name, classification_report, cm_path, roc_path, metrics):
        """
        Crea un informe PDF consolidado con gráficos y métricas.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Página principal
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Informe del Modelo: {model_name}", ln=True, align='C')

        # Métricas
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Métricas principales:", ln=True, align='L')
        for metric, value in metrics.items():
            try:
                # Intentar formatear como número decimal
                pdf.cell(200, 10, txt=f"{metric}: {float(value):.2f}", ln=True, align='L')
            except ValueError:
                # Si el valor no es numérico, imprimirlo como está
                pdf.cell(200, 10, txt=f"{metric}: {value}", ln=True, align='L')

        # Reporte de clasificación
        pdf.cell(200, 10, txt="Reporte de Clasificación:", ln=True, align='L')
        pdf.set_font("Courier", size=8)
        for line in classification_report.split("\n"):
            pdf.cell(200, 5, txt=line, ln=True)

        # Matriz de Confusión
        pdf.add_page()
        pdf.cell(200, 10, txt="Matriz de Confusión:", ln=True, align='L')
        pdf.image(cm_path, x=10, y=30, w=180)

        # Curva ROC
        if roc_path:
            pdf.add_page()
            pdf.cell(200, 10, txt="Curva ROC:", ln=True, align='L')
            pdf.image(roc_path, x=10, y=30, w=180)

        # Guardar PDF
        pdf_path = os.path.join(self.report_dir, f"{model_name}_report.pdf")
        pdf.output(pdf_path)
        print(f"Informe PDF generado: {pdf_path}")
        return pdf_path

    def plot_confusion_matrix_manual(self, model, X_val, y_val, model_name, report_dir):

        """
           Genera y guarda la matriz de confusión manualmente.
           """
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        path = os.path.join(self.report_dir, f"{model_name}_confusion_matrix_manual.png")
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.savefig(path)
        plt.close()
        return path
