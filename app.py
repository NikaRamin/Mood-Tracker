import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from hugging_predictor import HuggingFaceMoodPredictor

# Initialize HuggingFace predictor
hugging_predictor = None
try:
    hugging_predictor = HuggingFaceMoodPredictor()
except Exception as e:
    print(f"Error initializing HuggingFace predictor: {str(e)}")

def track_mood():
    mood = mood_entry.get()
    if mood == "Enter your mood here..." or not mood.strip():
        messagebox.showwarning("Input Required", "Please enter your mood first!")
        return
    
    # Show the result label only when there's input
    result_label.pack(pady=20, fill=tk.BOTH, expand=True)
    
    result_text = ""
    sentiment = "neutral"
    
    # Get HuggingFace prediction and suggestions
    if hugging_predictor:
        try:
            # Get mood analysis
            hf_result = hugging_predictor.analyze_mood(mood)
            result_label['foreground'] = 'black'
            if hf_result:
                sentiment = hf_result['mood']  # Save sentiment for CSV
                result_text += "━━━ HuggingFace Analysis ━━━\n"
                result_text += f"Detected Mood: {hf_result['mood'].title()}\n"
                result_text += f"Confidence: {hf_result['confidence']:.2%}\n"
                
                # Get therapist response
                therapist_msg = hugging_predictor.therapist_response(hf_result['mood'], mood)
                if therapist_msg:
                    result_text += "💌 Therapist's Response 💌\n"
                    result_text += "━━━━━━━━━━━━━━━━━━━━━━\n"
                    result_text += f"{therapist_msg}\n"
                    result_text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                
                # Get therapeutic suggestions
                suggestions = hugging_predictor.suggest_activities(hf_result['mood'], mood)
                if suggestions:
                    result_text += "\n✨ Therapeutic Insights ✨\n"
                    result_text += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    emojis = ["🌟", "⭐", "✨"]  # Different star emojis for each activity
                    for i, activity in enumerate(suggestions['activities'], 1):
                        result_text += f"  {emojis[(i-1) % 3]} {activity}\n"
                result_text += "\n━━━━━━━━━━━━━━━━━━━━━━"
                
                # Save to CSV
                save_mood_data(mood, sentiment)
        except Exception as e:
            result_text += "HuggingFace Analysis: Temporarily unavailable\n"
            print(f"Error: {str(e)}")
    else:
        result_text += "Error: HuggingFace service is not available. Please check your API key.\n"
    
    # Update the display
    result_label['text'] = result_text
    
    # Clear the entry field and reset to placeholder
    mood_entry.delete(0, tk.END)
    mood_entry.insert(0, "Enter your mood here...")
    mood_entry.config(fg='gray')

def analyze_mood_patterns():
    """Analyze mood patterns and display results"""
    try:
        df = pd.read_csv('mood_data.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Add day of week and time of day
        df['DayOfWeek'] = df['Timestamp'].dt.day_name()
        df['Hour'] = df['Timestamp'].dt.hour
        df['TimeOfDay'] = df['Hour'].apply(lambda x: 
            'Morning' if 5 <= x < 12 else
            'Afternoon' if 12 <= x < 17 else
            'Evening' if 17 <= x < 22 else
            'Night'
        )
        
        # Analyze happiest day
        day_mood_counts = pd.crosstab(df['DayOfWeek'], df['Sentiment'])
        if 'happy' in day_mood_counts.columns:
            happiest_day = day_mood_counts['happy'].idxmax()
            happy_count = day_mood_counts.loc[happiest_day, 'happy']
        else:
            happiest_day = "Not enough data"
            happy_count = 0
            
        # Analyze mood by time of day
        time_mood_counts = pd.crosstab(df['TimeOfDay'], df['Sentiment'])
        
        # Find happiest and saddest times
        if len(time_mood_counts) > 0:
            if 'happy' in time_mood_counts.columns:
                happiest_time = time_mood_counts['happy'].idxmax()
                happy_time_count = time_mood_counts.loc[happiest_time, 'happy']
            else:
                happiest_time = "Not enough data"
                happy_time_count = 0
                
            if 'sad' in time_mood_counts.columns:
                saddest_time = time_mood_counts['sad'].idxmax()
                sad_time_count = time_mood_counts.loc[saddest_time, 'sad']
            else:
                saddest_time = "Not enough data"
                sad_time_count = 0
        else:
            happiest_time = saddest_time = "Not enough data"
            happy_time_count = sad_time_count = 0
            
        # Create analysis window
        analysis_window = tk.Toplevel(window)
        analysis_window.title("Mood Pattern Analysis")
        analysis_window.geometry("1200x1200")

        # Handle window close event
        def on_window_close():
            plt.close('all')
            analysis_window.destroy()
        
        analysis_window.protocol("WM_DELETE_WINDOW", on_window_close)
        
        # Configure style
        style = ttk.Style()
        style.configure('Analysis.TLabel', font=('Arial', 14), padding=8)
        
        # Create frames
        header_frame = ttk.Frame(analysis_window)
        header_frame.pack(pady=20)
        
        stats_frame = ttk.Frame(analysis_window)
        stats_frame.pack(pady=15, padx=30, fill=tk.X)
        
        # Display title
        ttk.Label(
            header_frame,
            text="Mood Pattern Analysis",
            style='Analysis.TLabel'
        ).pack()
        
        # Display statistics
        ttk.Label(
            stats_frame,
            text=f"Most Happy Day: {happiest_day} (Count: {happy_count})",
            style='Analysis.TLabel'
        ).pack(pady=8)
        
        ttk.Label(
            stats_frame,
            text=f"Happiest Time: {happiest_time} (Count: {happy_time_count})",
            style='Analysis.TLabel'
        ).pack(pady=8)
        
        ttk.Label(
            stats_frame,
            text=f"Most Sad Time: {saddest_time} (Count: {sad_time_count})",
            style='Analysis.TLabel'
        ).pack(pady=8)
        
        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
        
        # Plot mood distribution by day
        day_mood_counts.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Mood Distribution by Day of Week', pad=20, fontsize=16)
        ax1.set_xlabel('Day of Week', fontsize=14)
        ax1.set_ylabel('Count', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot mood distribution by time of day
        time_mood_counts.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Mood Distribution by Time of Day', pad=20, fontsize=16)
        ax2.set_xlabel('Time of Day', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=4.0)
        
        # Create canvas frame
        canvas_frame = ttk.Frame(analysis_window)
        canvas_frame.pack(pady=25, padx=30, expand=True, fill=tk.BOTH)
        
        # Embed plot
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        
    except FileNotFoundError:
        messagebox.showwarning("No Data", "No mood data available for analysis.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        plt.close('all')

def save_mood_data(mood, sentiment):
    try:
        data = pd.read_csv("mood_data.csv", parse_dates=['Timestamp'])
    except FileNotFoundError:
        data = pd.DataFrame(columns=["Timestamp", "Mood", "Sentiment"])
    
    new_data = pd.DataFrame({
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Mood': [mood], 
        'Sentiment': [sentiment]
    })
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv('mood_data.csv', index=False)

# Configure the main window
window = tk.Tk()
window.title("Mood Tracker")
window.configure(bg='#f0f0f0')

main_frame = ttk.Frame(window, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Style configuration
style = ttk.Style()
style.configure('TFrame', background='#f0f0f0')
style.configure('TButton', padding=6, font=('Arial', 12))
style.configure('TLabel', background='#f0f0f0', font=('Arial', 16))

# Title
title_label = ttk.Label(main_frame, text="AI-Powered Mood Tracker", font=('Arial', 24, 'bold'))
title_label.pack(pady=(0, 30))

# Mood input section
mood_label = ttk.Label(main_frame, text="How are you feeling today?")
mood_label.pack(pady=(0, 10))

mood_entry = tk.Entry(main_frame, width=40, font=('Arial', 12))
mood_entry.insert(0, "Enter your mood here...")
mood_entry.config(fg='gray', bg='white')
mood_entry.pack(pady=(0, 20))

# Entry field events
def clear_entry(event):
    if mood_entry.get() == "Enter your mood here...":
        mood_entry.delete(0, tk.END)
        mood_entry.config(fg='black')

def on_focus_out(event):
    if mood_entry.get() == "":
        mood_entry.insert(0, "Enter your mood here...")
        mood_entry.config(fg='gray')

mood_entry.bind('<FocusIn>', clear_entry)
mood_entry.bind('<FocusOut>', on_focus_out)

# Buttons
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=20)

track_button = ttk.Button(button_frame, text="Save Mood", command=track_mood)
track_button.pack(side=tk.LEFT, padx=10)

analyze_button = ttk.Button(button_frame, text="Analyze Patterns", command=analyze_mood_patterns)
analyze_button.pack(side=tk.LEFT, padx=10)

# Result label - initially hidden
result_label = tk.Label(
    main_frame, 
    text="", 
    font=('Arial', 12), 
    bg='#f0f0f0',  # Match the background color
    fg='black', 
    wraplength=400
)
# Don't pack the result label initially

# Window configuration
window.geometry("500x700")
window.mainloop()

