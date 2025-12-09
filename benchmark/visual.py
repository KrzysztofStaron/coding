import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['non-coding', 'my-model', 'coding']
scores = [0.23780487804878048, None, 0.4756097560975609]
colors = ['#4A90E2', '#808080', '#50C878']

# Estimate your model's score (between Model1 and Model2)
estimated_score = (scores[0] + scores[2]) / 2  # Middle point
scores[1] = estimated_score

# Create figure with better styling
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#1E1E1E')
ax.set_facecolor('#2D2D2D')

# Create bar chart
bars = []
for i, (model, score, color) in enumerate(zip(models, scores, colors)):
    if i == 1:  # Your model - special styling
        bar = ax.bar(i, score, color=color, edgecolor='white', linewidth=2, 
                     alpha=0.6, width=0.6, hatch='///', linestyle='--')
        # Add question marks
        ax.text(bar[0].get_x() + bar[0].get_width()/2., score/2,
                '???', ha='center', va='center', fontsize=24, fontweight='bold', 
                color='white', alpha=0.8)
        ax.text(bar[0].get_x() + bar[0].get_width()/2., score + 0.01,
                '?', ha='center', va='bottom', fontsize=18, fontweight='bold', 
                color='white', alpha=0.9)
    else:
        bar = ax.bar(i, score, color=color, edgecolor='white', linewidth=2, 
                     alpha=0.9, width=0.6)
        # Add value labels
        ax.text(bar[0].get_x() + bar[0].get_width()/2., score + 0.01,
                f'{score:.2%}',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')
    bars.append(bar)

# Customize axes
ax.set_ylabel('Pass@1 Score', fontsize=14, fontweight='bold', color='white')
ax.set_title('HumanEval Benchmark Results\n(0.5B Parameter Models)', 
             fontsize=16, fontweight='bold', color='white', pad=20)
ax.set_ylim(0, max(scores) * 1.2)

# Format y-axis as percentage
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.tick_params(colors='white', labelsize=12)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=14, fontweight='bold', color='white')

# Remove X axis line
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)  # Remove X axis
ax.spines['left'].set_color('white')
ax.grid(axis='y', alpha=0.3, color='white', linestyle='--', linewidth=0.5)

# Add prediction arrows
# Arrow from Model1 to Your Model
ax.annotate('Expected\nimprovement',
            xy=(1, scores[1]), xytext=(0, scores[0] + 0.1),
            arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2, alpha=0.7),
            fontsize=10, color='#FFD700', fontweight='bold',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#2D2D2D', edgecolor='#FFD700', linewidth=1.5, alpha=0.8))

# Arrow from Your Model to Model2 (showing it's better but not as good)
ax.annotate('Target',
            xy=(2, scores[2]), xytext=(1, scores[1] + 0.1),
            arrowprops=dict(arrowstyle='->', color='#50C878', lw=2, alpha=0.5, linestyle='--'),
            fontsize=10, color='#50C878', fontweight='bold',
            ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#2D2D2D', edgecolor='#50C878', linewidth=1.5, alpha=0.6))

plt.tight_layout()
plt.savefig('humaneval_results.png', dpi=300, facecolor='#1E1E1E', bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print("HumanEval Benchmark Results")
print(f"{'='*60}")
print(f"Qwen2.5-0.5B-Instruct:      {scores[0]:.2%} ({scores[0]*164:.0f}/164 problems)")
print(f"My Model (Predicted):       ??? (between {scores[0]:.2%} and {scores[2]:.2%})")
print(f"Qwen2.5-0.5B-Coder-Instruct: {scores[2]:.2%} ({scores[2]*164:.0f}/164 problems)")
print(f"\nTarget: Better than Instruct ({scores[0]:.2%}), approaching Coder ({scores[2]:.2%})")
print(f"{'='*60}")
