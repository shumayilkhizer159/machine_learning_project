"""
Exploratory Data Analysis for NFL Big Data Bowl 2026
This script performs comprehensive EDA on the tracking data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_sample_data(data_dir='/home/ubuntu/nfl_project/data'):
    """Load a sample of the data for EDA."""
    train_dir = Path(data_dir) / 'train'
    
    # Load first week as sample
    input_df = pd.read_csv(train_dir / 'input_2023_w01.csv')
    output_df = pd.read_csv(train_dir / 'output_2023_w01.csv')
    
    return input_df, output_df

def basic_statistics(input_df, output_df):
    """Generate basic statistics about the data."""
    print("=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    
    print("\n--- Input Data ---")
    print(f"Shape: {input_df.shape}")
    print(f"Columns: {input_df.columns.tolist()}")
    print(f"\nUnique games: {input_df['game_id'].nunique()}")
    print(f"Unique plays: {input_df['play_id'].nunique()}")
    print(f"Unique players: {input_df['nfl_id'].nunique()}")
    print(f"Total frames: {input_df['frame_id'].max()}")
    
    print("\n--- Output Data ---")
    print(f"Shape: {output_df.shape}")
    print(f"Unique games: {output_df['game_id'].nunique()}")
    print(f"Unique plays: {output_df['play_id'].nunique()}")
    print(f"Unique players: {output_df['nfl_id'].nunique()}")
    
    print("\n--- Players to Predict ---")
    if 'player_to_predict' in input_df.columns:
        print(f"Total players: {len(input_df)}")
        print(f"Players to predict: {input_df['player_to_predict'].sum()}")
        print(f"Percentage: {input_df['player_to_predict'].mean() * 100:.2f}%")
    
    print("\n--- Frame Statistics ---")
    if 'num_frames_output' in input_df.columns:
        print(f"Min frames to predict: {input_df['num_frames_output'].min()}")
        print(f"Max frames to predict: {input_df['num_frames_output'].max()}")
        print(f"Mean frames to predict: {input_df['num_frames_output'].mean():.2f}")
        print(f"Median frames to predict: {input_df['num_frames_output'].median():.2f}")
    
    return

def analyze_player_positions(input_df):
    """Analyze player positions and roles."""
    print("\n" + "=" * 80)
    print("PLAYER ANALYSIS")
    print("=" * 80)
    
    print("\n--- Player Positions ---")
    if 'player_position' in input_df.columns:
        pos_counts = input_df['player_position'].value_counts()
        print(pos_counts.head(10))
    
    print("\n--- Player Roles ---")
    if 'player_role' in input_df.columns:
        role_counts = input_df['player_role'].value_counts()
        print(role_counts)
    
    print("\n--- Player Side ---")
    if 'player_side' in input_df.columns:
        side_counts = input_df['player_side'].value_counts()
        print(side_counts)
    
    return

def analyze_tracking_data(input_df):
    """Analyze tracking data statistics."""
    print("\n" + "=" * 80)
    print("TRACKING DATA ANALYSIS")
    print("=" * 80)
    
    print("\n--- Position Statistics ---")
    print(f"X coordinate range: [{input_df['x'].min():.2f}, {input_df['x'].max():.2f}]")
    print(f"Y coordinate range: [{input_df['y'].min():.2f}, {input_df['y'].max():.2f}]")
    
    print("\n--- Speed Statistics ---")
    print(f"Speed (s) - Mean: {input_df['s'].mean():.2f}, Std: {input_df['s'].std():.2f}")
    print(f"Speed (s) - Min: {input_df['s'].min():.2f}, Max: {input_df['s'].max():.2f}")
    
    print("\n--- Acceleration Statistics ---")
    print(f"Acceleration (a) - Mean: {input_df['a'].mean():.2f}, Std: {input_df['a'].std():.2f}")
    print(f"Acceleration (a) - Min: {input_df['a'].min():.2f}, Max: {input_df['a'].max():.2f}")
    
    print("\n--- Direction Statistics ---")
    print(f"Direction (dir) - Mean: {input_df['dir'].mean():.2f}, Std: {input_df['dir'].std():.2f}")
    print(f"Orientation (o) - Mean: {input_df['o'].mean():.2f}, Std: {input_df['o'].std():.2f}")
    
    print("\n--- Ball Landing Location ---")
    if 'ball_land_x' in input_df.columns:
        print(f"Ball land X - Mean: {input_df['ball_land_x'].mean():.2f}, Std: {input_df['ball_land_x'].std():.2f}")
        print(f"Ball land Y - Mean: {input_df['ball_land_y'].mean():.2f}, Std: {input_df['ball_land_y'].std():.2f}")
    
    return

def create_visualizations(input_df, output_df, save_dir='/home/ubuntu/nfl_project/figures'):
    """Create visualizations for EDA."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Distribution of frames to predict
    if 'num_frames_output' in input_df.columns:
        plt.figure(figsize=(10, 6))
        input_df['num_frames_output'].hist(bins=50, edgecolor='black')
        plt.xlabel('Number of Frames to Predict')
        plt.ylabel('Frequency')
        plt.title('Distribution of Output Frame Counts')
        plt.savefig(save_dir / 'frames_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: frames_distribution.png")
    
    # 2. Speed distribution
    plt.figure(figsize=(10, 6))
    plt.hist(input_df['s'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Speed (yards/second)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Player Speed')
    plt.savefig(save_dir / 'speed_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: speed_distribution.png")
    
    # 3. Player positions heatmap
    plt.figure(figsize=(12, 8))
    sample = input_df.sample(min(10000, len(input_df)))
    plt.hexbin(sample['x'], sample['y'], gridsize=30, cmap='YlOrRd')
    plt.colorbar(label='Count')
    plt.xlabel('X Position (yards)')
    plt.ylabel('Y Position (yards)')
    plt.title('Player Position Heatmap (Sample)')
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.savefig(save_dir / 'position_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: position_heatmap.png")
    
    # 4. Ball landing locations
    if 'ball_land_x' in input_df.columns:
        plt.figure(figsize=(12, 8))
        unique_plays = input_df.drop_duplicates(subset=['game_id', 'play_id'])
        plt.scatter(unique_plays['ball_land_x'], unique_plays['ball_land_y'], 
                   alpha=0.5, s=20)
        plt.xlabel('Ball Landing X (yards)')
        plt.ylabel('Ball Landing Y (yards)')
        plt.title('Ball Landing Locations')
        plt.xlim(0, 120)
        plt.ylim(0, 53.3)
        plt.savefig(save_dir / 'ball_landing_locations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: ball_landing_locations.png")
    
    # 5. Player roles distribution
    if 'player_role' in input_df.columns:
        plt.figure(figsize=(10, 6))
        role_counts = input_df['player_role'].value_counts()
        role_counts.plot(kind='bar', edgecolor='black')
        plt.xlabel('Player Role')
        plt.ylabel('Count')
        plt.title('Distribution of Player Roles')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / 'player_roles.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: player_roles.png")
    
    # 6. Speed by player role
    if 'player_role' in input_df.columns:
        plt.figure(figsize=(12, 6))
        input_df.boxplot(column='s', by='player_role', figsize=(12, 6))
        plt.xlabel('Player Role')
        plt.ylabel('Speed (yards/second)')
        plt.title('Speed Distribution by Player Role')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / 'speed_by_role.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: speed_by_role.png")
    
    return

def analyze_sample_play(input_df, output_df, save_dir='/home/ubuntu/nfl_project/figures'):
    """Visualize a sample play trajectory."""
    save_dir = Path(save_dir)
    
    # Get a sample play
    sample_game = input_df['game_id'].iloc[0]
    sample_play = input_df['play_id'].iloc[0]
    
    play_input = input_df[(input_df['game_id'] == sample_game) & 
                          (input_df['play_id'] == sample_play)]
    play_output = output_df[(output_df['game_id'] == sample_game) & 
                            (output_df['play_id'] == sample_play)]
    
    print("\n" + "=" * 80)
    print(f"SAMPLE PLAY ANALYSIS (Game: {sample_game}, Play: {sample_play})")
    print("=" * 80)
    print(f"Players in play: {play_input['nfl_id'].nunique()}")
    print(f"Input frames: {play_input['frame_id'].max()}")
    print(f"Output frames: {play_output['frame_id'].max()}")
    
    # Visualize trajectories
    plt.figure(figsize=(14, 8))
    
    # Draw field
    plt.axhline(y=0, color='black', linewidth=2)
    plt.axhline(y=53.3, color='black', linewidth=2)
    plt.axvline(x=0, color='black', linewidth=2)
    plt.axvline(x=120, color='black', linewidth=2)
    
    # Plot ball landing location
    if 'ball_land_x' in play_input.columns:
        ball_x = play_input['ball_land_x'].iloc[0]
        ball_y = play_input['ball_land_y'].iloc[0]
        plt.scatter(ball_x, ball_y, s=200, c='red', marker='*', 
                   label='Ball Landing', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot player trajectories
    for nfl_id in play_input['nfl_id'].unique():
        player_input = play_input[play_input['nfl_id'] == nfl_id].sort_values('frame_id')
        player_output = play_output[play_output['nfl_id'] == nfl_id].sort_values('frame_id')
        
        # Get player info
        player_role = player_input['player_role'].iloc[0] if 'player_role' in player_input.columns else 'Unknown'
        
        # Plot input trajectory (before pass)
        plt.plot(player_input['x'], player_input['y'], 
                'o-', alpha=0.6, linewidth=2, markersize=4, label=f'{player_role} (input)')
        
        # Plot output trajectory (after pass)
        if len(player_output) > 0:
            plt.plot(player_output['x'], player_output['y'], 
                    's--', alpha=0.8, linewidth=2, markersize=4, label=f'{player_role} (output)')
    
    plt.xlabel('X Position (yards)')
    plt.ylabel('Y Position (yards)')
    plt.title(f'Sample Play Trajectories (Game: {sample_game}, Play: {sample_play})')
    plt.xlim(0, 120)
    plt.ylim(0, 53.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_play_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: sample_play_trajectories.png")
    
    return

def main():
    """Main EDA function."""
    print("\n" + "=" * 80)
    print("NFL BIG DATA BOWL 2026 - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading sample data (Week 1)...")
    input_df, output_df = load_sample_data()
    print("✓ Data loaded successfully")
    
    # Run analyses
    basic_statistics(input_df, output_df)
    analyze_player_positions(input_df)
    analyze_tracking_data(input_df)
    
    # Create visualizations
    create_visualizations(input_df, output_df)
    
    # Analyze sample play
    analyze_sample_play(input_df, output_df)
    
    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print("\nAll visualizations saved to: /home/ubuntu/nfl_project/figures/")
    
    return input_df, output_df

if __name__ == '__main__':
    input_df, output_df = main()
