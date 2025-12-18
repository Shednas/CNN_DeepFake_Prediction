"""
Professional Deepfake Detection Visualization Module
======================================================
Handles image prediction, analysis, and professional visualization dashboard
for deepfake detection research.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.stats import entropy
import cv2
import glob
from datetime import datetime


# ===== DEVICE CONFIGURATION =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== MODEL LOADING =====
MODEL_PATH = 'CNN_DeepFake_Prediction/model/cnn_model.pth'
model = None
classes = ['AI-Generated', 'Real']

# ===== IMAGE TRANSFORM =====
img_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_model():
    """Load the CNN model for predictions."""
    global model
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script'))
    from cnn_model import SimpleCNN  # type: ignore
    
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


def analyze_image_characteristics(image_path):
    """
    Analyze image characteristics for explainability.
    
    Returns dict with: contrast, edge_magnitude, color_entropy, mean_color
    """
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Contrast calculation
    contrast = np.std(img_gray)
    
    # Edge detection
    edges_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2).mean()
    
    # Color entropy
    hist_r = np.histogram(img_array[:, :, 0], bins=256)[0]
    hist_g = np.histogram(img_array[:, :, 1], bins=256)[0]
    hist_b = np.histogram(img_array[:, :, 2], bins=256)[0]
    
    hist_r = hist_r / hist_r.sum()
    hist_g = hist_g / hist_g.sum()
    hist_b = hist_b / hist_b.sum()
    
    color_entropy = (entropy(hist_r) + entropy(hist_g) + entropy(hist_b)) / 3
    
    # Mean color
    mean_color = img_array.mean(axis=(0, 1))
    
    return {
        'contrast': contrast,
        'edge_magnitude': edge_magnitude,
        'color_entropy': color_entropy,
        'mean_color': mean_color
    }


def generate_professional_report(image_path, predicted_class, confidence, all_probs, analysis):
    """Generate a professionally written text report."""
    
    report = f"""
{'='*70}
DEEPFAKE DETECTION ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

IMAGE INFORMATION
-----------------
File: {os.path.basename(image_path)}
Size: {Image.open(image_path).size}

PREDICTION RESULT
-----------------
Classification: {predicted_class.upper()}
Confidence: {confidence:.2%}

Probability Breakdown:
  AI-Generated: {all_probs[0]:.2%}
  Real Image:   {all_probs[1]:.2%}

CONFIDENCE ANALYSIS
-------------------
"""
    
    if confidence > 0.85:
        report += f"Level: VERY HIGH (>85%)\n"
        report += f"Assessment: Strong evidence for {predicted_class}\n"
    elif confidence > 0.70:
        report += f"Level: HIGH (70-85%)\n"
        report += f"Assessment: Good evidence for {predicted_class}\n"
    elif confidence > 0.55:
        report += f"Level: MODERATE (55-70%)\n"
        report += f"Assessment: Moderate evidence for {predicted_class}\n"
    else:
        report += f"Level: LOW (<55%)\n"
        report += f"Assessment: Weak evidence, borderline case\n"
    
    report += f"""
IMAGE CHARACTERISTICS
---------------------
Contrast Level: {analysis['contrast']:.2f} 
  {'(High)' if analysis['contrast'] > 60 else '(Medium)' if analysis['contrast'] > 40 else '(Low)'}

Edge Definition: {analysis['edge_magnitude']:.3f}
  {'(Sharp)' if analysis['edge_magnitude'] > 5 else '(Moderate)' if analysis['edge_magnitude'] > 2 else '(Blurry)'}

Color Entropy: {analysis['color_entropy']:.3f}
  {'(Diverse)' if analysis['color_entropy'] > 4 else '(Moderate)' if analysis['color_entropy'] > 2 else '(Limited)'}

Mean Color (RGB):
  R={int(analysis['mean_color'][0])}, G={int(analysis['mean_color'][1])}, B={int(analysis['mean_color'][2])}

MODEL INTERPRETATION
--------------------
"""
    
    if predicted_class == 'AI-Generated':
        report += """The model detected patterns consistent with AI generation:
- Unusual texture distributions
- Pixel-level artifacts typical of generative models
- Statistical anomalies in color channels
- Potential blending artifacts

Recommendation: Review with caution - likely synthetic content
"""
    else:
        report += """The model detected natural image characteristics:
- Realistic noise patterns
- Natural color gradients
- Coherent lighting physics
- Expected statistical properties

Recommendation: Image appears authentic
"""
    
    report += f"""
METHODOLOGY NOTES
-----------------
Model: Convolutional Neural Network (CNN)
Architecture: 3-layer CNN with 128x128 RGB input
Training Data: Balanced dataset of real and AI-generated images

DISCLAIMER
----------
This analysis is based on machine learning patterns and should not be
considered definitive proof. For critical applications, combine with
manual review and additional verification methods.
AI detection accuracy depends on training data quality and may have
limitations on novel or out-of-distribution images.
{'='*70}
"""
    
    return report


def create_visualization_dashboard(image, predicted_class, confidence, all_probs, analysis):
    """Create professional visualization dashboard."""
    
    # Color scheme
    primary_color = '#e74c3c' if predicted_class == 'AI-Generated' else '#27ae60'
    secondary_color = '#ecf0f1' if predicted_class == 'AI-Generated' else '#d5f4e6'
    
    # Create figure
    fig = plt.figure(figsize=(22, 14), facecolor='#f8f9fa')
    
    # Custom grid layout
    gs = fig.add_gridspec(4, 4, height_ratios=[0.35, 1.5, 1.2, 1.2], 
                         hspace=0.5, wspace=0.4, 
                         left=0.06, right=0.97, top=0.96, bottom=0.08)
    
    # ===== LARGE TITLE AT TOP =====
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'DEEPFAKE DETECTION ANALYSIS DASHBOARD', 
                 transform=ax_title.transAxes, fontsize=36, fontweight='bold', 
                 ha='center', va='center', color='#1a1a1a')
    
    # ===== ROW 1: INPUT IMAGE + LARGE PREDICTION BOX =====
    
    # Input Image (left, smaller)
    ax_input = fig.add_subplot(gs[1, 0:2])
    ax_input.imshow(image)
    ax_input.axis('off')
    ax_input.set_title('Input Image', fontsize=13, fontweight='bold', pad=12, color='#2c3e50')
    ax_input.set_facecolor('#ffffff')
    for spine in ax_input.spines.values():
        spine.set_edgecolor('#34495e')
        spine.set_linewidth(2)
        spine.set_visible(True)
    
    # Prediction (right, large)
    ax_pred = fig.add_subplot(gs[1, 2:])
    ax_pred.axis('off')
    ax_pred.set_facecolor('#ffffff')
    
    # Large prediction box
    rect = plt.Rectangle((0.08, 0.1), 0.84, 0.8, transform=ax_pred.transAxes,
                         facecolor=secondary_color, edgecolor=primary_color, 
                         linewidth=4, alpha=0.95)
    ax_pred.add_patch(rect)
    
    # Prediction text
    pred_text = predicted_class.upper().replace('_', ' ')
    ax_pred.text(0.5, 0.80, 'PREDICTION', transform=ax_pred.transAxes,
                ha='center', va='center', fontsize=13, color='#7f8c8d', fontweight='bold')
    ax_pred.text(0.5, 0.58, pred_text, transform=ax_pred.transAxes,
                ha='center', va='center', fontsize=52, color=primary_color, fontweight='bold')
    ax_pred.text(0.5, 0.30, f'{confidence:.1%}', transform=ax_pred.transAxes,
                ha='center', va='center', fontsize=36, color=primary_color, fontweight='bold')
    ax_pred.text(0.5, 0.12, 'CONFIDENCE', transform=ax_pred.transAxes,
                ha='center', va='center', fontsize=11, color='#95a5a6', fontweight='bold')
    
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(0, 1)
    
    # ===== ROW 2: HEATMAPS AND HISTOGRAM =====
    
    # Intensity Heatmap
    ax_heatmap = fig.add_subplot(gs[2, 0:2])
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    im_heat = ax_heatmap.imshow(img_gray, cmap='hot', interpolation='bilinear')
    ax_heatmap.axis('off')
    ax_heatmap.set_title('Intensity Heatmap', fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax_heatmap.set_facecolor('#ffffff')
    cbar_heat = plt.colorbar(im_heat, ax=ax_heatmap, fraction=0.08, pad=0.05, shrink=0.9)
    cbar_heat.set_label('Intensity', color='#2c3e50', fontweight='bold', fontsize=9)
    cbar_heat.ax.tick_params(labelsize=8)
    
    # Edge Detection
    ax_edge = fig.add_subplot(gs[2, 2])
    edges = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
    edges_norm = np.sqrt(edges**2)
    im_edge = ax_edge.imshow(edges_norm, cmap='viridis', interpolation='bilinear')
    ax_edge.axis('off')
    ax_edge.set_title('Edge Detection', fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax_edge.set_facecolor('#ffffff')
    cbar_edge = plt.colorbar(im_edge, ax=ax_edge, fraction=0.08, pad=0.05, shrink=0.9)
    cbar_edge.set_label('Edge', color='#2c3e50', fontweight='bold', fontsize=9)
    cbar_edge.ax.tick_params(labelsize=8)
    
    # RGB Distribution
    ax_rgb = fig.add_subplot(gs[2, 3])
    ax_rgb.set_facecolor('#ffffff')
    img_array = np.array(image)
    channels = ['R', 'G', 'B']
    colors_channels = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, (channel, color) in enumerate(zip(channels, colors_channels)):
        channel_data = img_array[:, :, i].flatten()
        ax_rgb.hist(channel_data, bins=20, alpha=0.65, label=channel, color=color, 
                   edgecolor='black', linewidth=0.7)
    
    ax_rgb.set_xlabel('Pixel Value', fontweight='bold', fontsize=9, color='#2c3e50')
    ax_rgb.set_ylabel('Frequency', fontweight='bold', fontsize=9, color='#2c3e50')
    ax_rgb.set_title('RGB Distribution', fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax_rgb.legend(loc='upper right', fontsize=8, framealpha=0.95)
    ax_rgb.grid(axis='y', alpha=0.2, color='#bdc3c7')
    ax_rgb.tick_params(labelsize=8)
    
    # ===== ROW 3: SCORES AND METRICS =====
    
    # Classification Scores
    ax_scores = fig.add_subplot(gs[3, 0:2])
    ax_scores.set_facecolor('#ffffff')
    class_labels = ['AI-Generated', 'Real Image']
    colors_bar = ['#e74c3c', '#27ae60']
    bars = ax_scores.barh(class_labels, all_probs, color=colors_bar, alpha=0.8, 
                          edgecolor='#2c3e50', linewidth=2)
    ax_scores.set_xlim(0, 1)
    ax_scores.set_xlabel('Probability', fontweight='bold', fontsize=11, color='#2c3e50')
    ax_scores.set_title('Classification Scores', fontsize=12, fontweight='bold', color='#2c3e50', pad=10)
    ax_scores.grid(axis='x', alpha=0.2, color='#bdc3c7')
    ax_scores.tick_params(labelsize=10)
    
    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        ax_scores.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontweight='bold', 
                      fontsize=11, color='#2c3e50')
    
    # Image Metrics
    ax_metrics = fig.add_subplot(gs[3, 2:])
    ax_metrics.axis('off')
    ax_metrics.set_facecolor('#ffffff')
    
    contrast_level = 'High' if analysis['contrast'] > 60 else 'Medium' if analysis['contrast'] > 40 else 'Low'
    edge_level = 'Sharp' if analysis['edge_magnitude'] > 5 else 'Moderate' if analysis['edge_magnitude'] > 2 else 'Blurry'
    entropy_level = 'Diverse' if analysis['color_entropy'] > 4 else 'Moderate' if analysis['color_entropy'] > 2 else 'Limited'
    
    metrics_text = f"""IMAGE ANALYSIS METRICS

Contrast:        {contrast_level:12} ({analysis['contrast']:6.2f})
Edge Definition: {edge_level:12} ({analysis['edge_magnitude']:6.3f})
Color Entropy:   {entropy_level:12} ({analysis['color_entropy']:6.3f})

Mean RGB Values:
  R={int(analysis['mean_color'][0]):3}  G={int(analysis['mean_color'][1]):3}  B={int(analysis['mean_color'][2]):3}

Confidence Level:
  {'VERY HIGH (>85%)' if confidence > 0.85 else 'HIGH (70-85%)' if confidence > 0.70 else 'MODERATE (55-70%)' if confidence > 0.55 else 'LOW (<55%)'}"""
    
    ax_metrics.text(0.08, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=10.5, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1.2', facecolor='#ffffff', alpha=0.95,
                            edgecolor=primary_color, linewidth=2.5),
                   color='#2c3e50', fontweight='bold')
    
    plt.draw()
    return fig


def predict_image(image_path):
    """
    Predict on a single image and generate professional output.
    
    Returns: (predicted_class, confidence)
    """
    global model
    
    if model is None:
        load_model()
    
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    img_tensor = img_transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
    
    predicted_idx = probs.argmax(dim=1).item()
    predicted_class = classes[predicted_idx]
    confidence = probs[0, predicted_idx].item()
    all_probs = probs[0].cpu().numpy()
    
    # Analyze image
    analysis = analyze_image_characteristics(image_path)
    
    # Generate report
    report = generate_professional_report(image_path, predicted_class, confidence, all_probs, analysis)
    
    # Save report
    os.makedirs('CNN_DeepFake_Prediction/results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'CNN_DeepFake_Prediction/results/report_{timestamp}.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Create and save visualization
    fig = create_visualization_dashboard(image, predicted_class, confidence, all_probs, analysis)
    dashboard_path = f'CNN_DeepFake_Prediction/results/dashboard_{timestamp}.png'
    fig.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    
    plt.close(fig)
    
    return predicted_class, confidence, report_path, dashboard_path


def find_image_path():
    """Find an image in toBePredicted folder."""
    
    # Check toBePredicted folder
    to_predict_images = glob.glob('CNN_DeepFake_Prediction/dataset/toBePredicted/*')
    if to_predict_images:
        return to_predict_images[0]
    
    # Fallback to test/real
    test_images = glob.glob('CNN_DeepFake_Prediction/dataset/test/real/*')
    if test_images:
        return test_images[0]
    
    return None


if __name__ == "__main__":
    image_path = find_image_path()
    
    if image_path is None:
        print("[ERROR] No images found in:")
        print("  - CNN_DeepFake_Prediction/dataset/toBePredicted/")
        print("  - CNN_DeepFake_Prediction/dataset/test/real/")
        exit(1)
    
    print(f"\n[PROCESSING] Image: {os.path.basename(image_path)}")
    
    try:
        predicted_class, confidence, report_path, dashboard_path = predict_image(image_path)
        
        print(f"\n[SUCCESS] Prediction: {predicted_class.upper()}")
        print(f"[SUCCESS] Confidence: {confidence:.1%}")
        print(f"\n[SAVED] Report: {report_path}")
        print(f"[SAVED] Dashboard: {dashboard_path}\n")
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {str(e)}\n")
        exit(1)
