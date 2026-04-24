import xml.etree.ElementTree as ET
from string_to_xml_to_vec import xml2vec, pretty_print_xml, linked_to_recursive
from plant_tokenizer import vec2token
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

class PlantSequenceVisualizer:
    """DNA-style plant architecture sequence visualization and comparison tool"""
        
    def __init__(self, gt_xml_path, pred_xml_path, gt_image_path, pred_image_path, diff_threshold=10):
        self.gt_xml_path = Path(gt_xml_path)
        self.pred_xml_path = Path(pred_xml_path)
        self.gt_image_path = Path(gt_image_path)
        self.pred_image_path = Path(pred_image_path)
        self.diff_threshold = diff_threshold
        
        self.token_config = {
            'chars': ['S', 'I', 'P', 'L', 'L', 'L'],
            'colors': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#f39c12', '#f39c12'],
            'param_color_range': (220, 100),
            'end_color': '#2c3e50',
            'gap_color': '#dddddd'
        }
        
        self.tokens_gt_orig, self.tokens_pred_orig = self._load_tokens()
        self.tokens_gt, self.tokens_pred = self._align_sequences(self.tokens_gt_orig, self.tokens_pred_orig)
        self.diff_indices, self.diff_offsets = self._find_significant_differences()
        
    def _load_xml_tokens(self, xml_path):
        if not xml_path.exists(): raise FileNotFoundError(f"XML file not found: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        root = linked_to_recursive(root)
        
        plant_instance_array = []
        # Support both old and new XML formats
        if len(root) > 0 and root[0].tag == 'plant':
            xml2vec(root[0], plant_instance_array)
        else:
            xml2vec(root, plant_instance_array)
        return vec2token(plant_instance_array)
    
    def _load_tokens(self):
        return self._load_xml_tokens(self.gt_xml_path), self._load_xml_tokens(self.pred_xml_path)
    
    def _find_significant_differences(self):
        diff_indices = []; diff_offsets = {}
        for i in range(len(self.tokens_gt)):
            gt_t = self.tokens_gt[i]; pred_t = self.tokens_pred[i]
            if gt_t == -1 and pred_t == -1: continue
            if gt_t == -1 or pred_t == -1:
                diff_indices.append(i)
                diff_offsets[i] = pred_t if gt_t == -1 else -gt_t
                continue
            if gt_t != pred_t:
                if gt_t < 24 or pred_t < 24:
                    diff_indices.append(i); diff_offsets[i] = int(pred_t) - int(gt_t)
                elif 24 <= gt_t < 240 and 24 <= pred_t < 240:
                    gt_p = int(gt_t) - 24; pred_p = int(pred_t) - 24
                    if gt_p != 0:
                        if abs((pred_p - gt_p) / gt_p) * 100 >= self.diff_threshold:
                            diff_indices.append(i); diff_offsets[i] = pred_p - gt_p
                    elif pred_p != 0:
                        diff_indices.append(i); diff_offsets[i] = pred_p - gt_p
                else:
                    diff_indices.append(i); diff_offsets[i] = int(pred_t) - int(gt_t)
        return diff_indices, diff_offsets
    
    def _align_sequences(self, tokens_gt, tokens_pred):
        GAP_TOKEN = -1; GAP_PENALTY = -1; MATCH_SCORE = 2; MISMATCH_PENALTY = -1
        n, m = len(tokens_gt), len(tokens_pred)
        score_matrix = np.zeros((n + 1, m + 1))
        for i in range(n + 1): score_matrix[i, 0] = GAP_PENALTY * i
        for j in range(m + 1): score_matrix[0, j] = GAP_PENALTY * j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score_matrix[i-1, j-1] + (MATCH_SCORE if tokens_gt[i-1] == tokens_pred[j-1] else MISMATCH_PENALTY)
                delete = score_matrix[i-1, j] + GAP_PENALTY
                insert = score_matrix[i, j-1] + GAP_PENALTY
                score_matrix[i, j] = max(match, delete, insert)
        aligned_gt = []; aligned_pred = []; i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + (MATCH_SCORE if tokens_gt[i-1] == tokens_pred[j-1] else MISMATCH_PENALTY):
                aligned_gt.insert(0, tokens_gt[i-1]); aligned_pred.insert(0, tokens_pred[j-1]); i -= 1; j -= 1
            elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + GAP_PENALTY:
                aligned_gt.insert(0, tokens_gt[i-1]); aligned_pred.insert(0, GAP_TOKEN); i -= 1
            else:
                aligned_gt.insert(0, GAP_TOKEN); aligned_pred.insert(0, tokens_pred[j-1]); j -= 1
        return aligned_gt, aligned_pred

    def _token_to_char_and_colors(self, token):
        if token == -1: return '-', self.token_config['gap_color'], 'black'
        elif token < 24:
            organ = token % 6
            return self.token_config['chars'][organ], self.token_config['colors'][organ], 'white'
        elif token < 240:
            param_value = token - 24; min_val, max_val = self.token_config['param_color_range']
            gray_value = int(min_val + (max_val - min_val) * (param_value / 215))
            return '', f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}', ('black' if gray_value > 128 else 'white')
        else: return '|', self.token_config['end_color'], 'white'
        
    def _create_dna_style_visualization(self, tokens, title, ax, show_differences=False, max_chars_per_line=120):
        ax.clear(); ax.set_xlim(0, max_chars_per_line + 1)
        num_lines = (len(tokens) + max_chars_per_line - 1) // max_chars_per_line
        ax.set_ylim(0, num_lines + 1); chars = []
        for line_idx in range(num_lines):
            start_idx = line_idx * max_chars_per_line; end_idx = min(start_idx + max_chars_per_line, len(tokens))
            y_pos = num_lines - line_idx
            for i, token in enumerate(tokens[start_idx:end_idx]):
                x_pos = i + 1; global_idx = start_idx + i
                char, bg_color, text_color = self._token_to_char_and_colors(token); chars.append(char)
                ax.add_patch(Rectangle((x_pos-0.5, y_pos-0.5), 1, 1, facecolor=bg_color, edgecolor='white', linewidth=0.5, zorder=1))
                ax.text(x_pos, y_pos, char, ha='center', va='center', fontsize=12, color=text_color, weight='bold', family='monospace', zorder=2)
                if show_differences and global_idx in self.diff_indices:
                    ax.add_patch(Rectangle((x_pos-0.5, y_pos-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2, zorder=3))
        ax.set_title(title, fontsize=14, weight='bold', pad=10); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        return chars
    
    def _load_and_display_image(self, image_path, ax, title):
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=14, weight='bold', pad=10); ax.axis('off'); return True
        ax.text(0.5, 0.5, f'{title}\n(Image not found)', ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); return False
            
    def render_to_axes(self, ax_img_gt, ax_img_pred, ax_seq_gt, ax_seq_pred, max_chars_per_line=80, title_suffix=''):
        """Render individual comparison components to externally provided axes"""
        self._load_and_display_image(self.gt_image_path, ax_img_gt, f"Input Image {title_suffix}")
        self._load_and_display_image(self.pred_image_path, ax_img_pred, f"Re-Rendered {title_suffix}")
        self._create_dna_style_visualization(self.tokens_gt, "", ax_seq_gt, show_differences=False, max_chars_per_line=max_chars_per_line)
        ax_seq_gt.set_title(f"GT Tokens {title_suffix}", fontsize=12, weight='bold')
        self._create_dna_style_visualization(self.tokens_pred, "", ax_seq_pred, show_differences=True, max_chars_per_line=max_chars_per_line)
        ax_seq_pred.set_title(f"Pred Tokens {title_suffix}", fontsize=12, weight='bold')
        significant_diffs = len(self.diff_indices); total_tokens = max(len(self.tokens_gt), len(self.tokens_pred))
        return {'accuracy': (1 - significant_diffs / total_tokens) * 100, 'diffs': significant_diffs}



if __name__ == '__main__':
    base_path = '../log/20250430_TrainValTestByPlotMoreData/dinov2-small_448_Sideview_gpt2-medium/results-80000/test_20250528'
    indices = [5, 10]  # Just 2 samples for quick verification
    fig, results = create_plant_comparison_grid(base_path, indices, cols=2)
    plt.savefig('plant_comparison_grid_preview.png', dpi=150, bbox_inches='tight')
    print('Preview saved to plant_comparison_grid_preview.png')
