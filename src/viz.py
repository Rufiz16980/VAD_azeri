import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def setup_style():
    """Applies a clean, modern aesthetic style to matplotlib plots."""
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.facecolor"] = "#f7f7f9"
    plt.rcParams["savefig.bbox"] = "tight"


def plot_tokenizer_diagnostic(tokens_per_word: List[int], output_path: str):
    """Plots a histogram of the tokens-per-word distribution for the tokenizer diagnostic."""
    setup_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure()
    mean_tokens = np.mean(tokens_per_word)
    median_tokens = np.median(tokens_per_word)
    
    plt.hist(tokens_per_word, bins=range(1, max(tokens_per_word) + 2), color="#4a90e2", edgecolor="black", alpha=0.8)
    plt.axvline(mean_tokens, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_tokens:.2f}")
    plt.axvline(median_tokens, color="green", linestyle="dotted", linewidth=1.5, label=f"Median: {median_tokens:.1f}")
    
    plt.title("Tokenizer Vocabulary Coverage (Tokens per Word)", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Tokens per Word")
    plt.ylabel("Word Count")
    plt.legend()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved tokenizer diagnostic plot to {output_path}")


def plot_segmentation_diagnostics(durations: List[float], confidences: List[float], output_dir: str):
    """Plots and saves histograms for segment durations and alignment confidence scores."""
    setup_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Segment Durations
    plt.figure()
    plt.hist(durations, bins=15, color="#50e3c2", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(durations), color="red", linestyle="dashed", label=f"Mean: {np.mean(durations):.2f}s")
    plt.title("Distribution of Derived Segment Durations", fontsize=14, fontweight="bold")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Segment Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "segment_durations.png"), dpi=300)
    plt.close()
    
    # 2. Alignment Confidence
    plt.figure()
    plt.hist(confidences, bins=15, color="#f5a623", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(confidences), color="red", linestyle="dashed", label=f"Mean: {np.mean(confidences):.4f}")
    plt.title("Distribution of Alignment Confidence Scores", fontsize=14, fontweight="bold")
    plt.xlabel("Alignment Confidence Score")
    plt.ylabel("Segment Count")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "alignment_confidence.png"), dpi=300)
    plt.close()
    print(f"Saved segmentation diagnostics (durations, confidence) to {output_dir}")


def plot_agreement_distribution(before_scores: List[float], after_scores: List[float], output_path: str):
    """Plots the cross-model agreement score distribution before and after filtering."""
    setup_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure()
    plt.hist(before_scores, bins=20, color="#d0021b", alpha=0.5, label="Raw Segments (Before Filtering)")
    plt.hist(after_scores, bins=20, color="#417505", alpha=0.6, label="Surviving Segments (After Filtering)")
    
    plt.title("Cross-Model Agreement Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Agreement Score (1 - Normalized Edit Distance)")
    plt.ylabel("Segment Count")
    plt.legend()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved agreement distribution plot to {output_path}")


def plot_waveform_alignment(
    waveform: np.ndarray, 
    sample_rate: int, 
    word_boundaries: List[Dict[str, Any]], 
    output_path: str,
    title: str = "Audio-Word Alignment Visualization"
):
    """Plots a waveform snippet with vertical lines representing aligned word boundaries."""
    setup_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    time_axis = np.arange(len(waveform)) / sample_rate
    
    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, waveform, color="#4a4a4a", alpha=0.6, label="Waveform")
    
    # Plot word boundary vertical lines
    for i, w in enumerate(word_boundaries):
        t_start = w["start_time"]
        t_end = w["end_time"]
        word = w["word"]
        
        # Color start boundary green, end boundary red
        plt.axvline(t_start, color="green", linestyle="--", alpha=0.7, linewidth=1.0)
        plt.axvline(t_end, color="red", linestyle=":", alpha=0.7, linewidth=1.0)
        
        # Annotate text near the middle of the word duration
        mid_time = (t_start + t_end) / 2.0
        plt.text(
            mid_time, 
            plt.ylim()[1] * 0.7, 
            word, 
            fontsize=8, 
            rotation=45, 
            ha="center", 
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )
        
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved waveform alignment plot to {output_path}")
