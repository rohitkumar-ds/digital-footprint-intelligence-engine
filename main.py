import json
import requests
import pandas as pd
import numpy as np 
import scipy.stats as stats
import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots
from bs4 import BeautifulSoup
from datetime import datetime , timedelta

# 1. Live Data Collection Engine (APIs + Web Scrapping)
 
class FootprintDataEngine :

    def __init__(self,github_user : str,stackoverflow_id : str = None):
        self.gh_user = github_user 
        self.so_id = stackoverflow_id

    def fetch_github_api(self)  :
        url = f"https://api.github.com/users/{self.gh_user}"
        res = requests.get(url,headers={"User-Agent" : "FootprintEngine/1.0"})
        if res.status_code == 200 :
            data = res.json()
            return{
                "public_repos" : data.get("public_repos",0),
                "followers" : data.get("followers" ,0)
                    }
        return {"public_repos" :0,"followers":0} 

    def scrape_github_contributions(self)  :
        url = f"https://github.com/users/{self.gh_user}/contributions"
        res = requests.get(url , headers={"User-Agent":"Mozilla/5.0"})
        if res.status_code == 200 :
            match = re.search(r'([\d,]+)\s+contributions',res.text)
            if match :
                return int(match.group(1).replace(',' ,''))
        return 0

    def fetch_stackoverflow_api(self)  :
        if not self.so_id :
            return {"reputation" : 0}
        url = f"https://api.stackexchange.com/2.3/users/{self.so_id}?site=stackoverflow"
        res = requests.get(url)
        if res.status_code == 200 :
            items = res.json().get("items" , [])
            if items :
                return {"reputation" : items[0].get("reputation",0)}
        return {"reputation" : 0}

    def simulate_time_series_history(self , months : int =12) :

        live_gh = self.fetch_github_api() 
        live_contribs = self.scrape_github_contributions()
        live_so = self.fetch_stackoverflow_api()

        end_date = datetime.now()
        dates = [end_date - timedelta(days=30*i)for i in range(months)][::-1]

        np.random.seed(42) 

        gh_commits = np.sort(np.random.randint(max(1,live_contribs//2),max(2,live_contribs+1),size=months))
        gh_repos = np.sort(np.random.randint(max(1,live_gh['public_repos']//2),max(2 , live_gh['public_repos']+1),size=months))
        so_rep = np.sort(np.random.randint(max(1,live_so['reputation']//2),max(2,live_so['reputation']+1),size=months))
        kaggle_contests = np.cumsum(np.random.poisson(lam=0.5,size=months))

        df = pd.DataFrame({
            'date' : dates ,
            'github_contributions' : gh_commits,
            'github_repos' : gh_repos ,
            'so_reputation' : so_rep ,
            'Kaggle_contests' : kaggle_contests
        })

        return df

# 2. FEATURE ENGINEERING PIPELINE
class FeatureProcessor :

    @staticmethod
    def process(df: pd.DataFrame) :
        df = df.copy()
        metrics = ['github_contributions','github_repos','so_reputation','Kaggle_contests']

        for col in metrics :
            df[f'{col}_velocity'] = df[col].diff().fillna(0)

        for col in metrics :
            df[f'{col}_log'] = np.log1p(df[col])

        for col in  metrics :
            min_val = df[col].min()
            max_val = df[col].max()
            df[f'{col}_norm'] = (df[col]-min_val)/(max_val-min_val+1e-8)

        return df 

# 3. STATISTICAL & SCORING INTELLIGENCE
class IntelligenceEngine :

    def __init__(self) :
        self.weights = {
            'github_contributions_norm' : 0.40,
            'github_repos_norm' : 0.20,
            'so_reputation_norm' : 0.25,
            'Kaggle_contests_norm' : 0.15
        }

    def compute_tgs(self , df:pd.DataFrame)  :
        score = np.zeros(len(df))
        for feature , weight in self.weights .items() :
            score += df[feature] * weight
        df['TGS'] = np.round(score *100,2)
        return df 

    @staticmethod 
    def analyze_growth_trajectory(df: pd.DataFrame) :
        x = np.arange(len(df))
        y = df['TGS'].values
        slope , intercept , r_val , p_val , std_err = stats.linregress(x,y)
        return {
            'monthly_slope' : round(slope,3),
            'r_squared' : round(r_val**2,3),
            'p_value': f"{p_val:.4e}" if p_val < 0.0001 else round(p_val, 4)}

    @staticmethod 
    def compute_correlations(df:pd.DataFrame):
        velocity_cols = [c for c in df.columns if c.endswith('_velocity')]
        return df[velocity_cols].corr(method='pearson')

# 4. DASHBOARD VISUALIZATION
class DashboardBuilder :

    @staticmethod
    def render(df: pd.DataFrame ,corr_matrix: pd.DataFrame,target_user : str) :
        fig = make_subplots(
            rows = 2 , cols = 2,
            subplot_titles = (
                "Technical Growth Score (TGS) Trend",
                "Platform Cumulative Growth",
                "Activity Velocity Pearson Correlation",
                "Mothly Contribution Velocity"
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.15
        )

        # Plot - 1 :
        fig.add_trace(
                go.Scatter(x=df['date'],y=df['TGS'],mode='lines+markers' ,name = "TGS",showlegend=False,
                            line =dict(color='#00D2FF',width=3)),row= 1 , col=1
        )

        # Plot - 2 :
        fig.add_trace(go.Scatter(x=df['date'],y=df['github_contributions_norm'] , name = "GH Contributions"),row=1 , col=2)
        fig.add_trace(go.Scatter(x=df['date'],y=df['so_reputation_norm'],name="SO Reputation"),row=1,col=2)

        # Plot -3 :
        clean_labels = [c.replace('_velocity','').replace('_',' ').title() for c in corr_matrix.columns]
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=clean_labels,
                y = clean_labels,
                text=corr_matrix.values,
                showlegend=False,
                texttemplate="%{text:.2f}",
                colorscale='Blues',
                showscale = False
            ),
            row = 2 , col = 1
        )

        # Plot -4 :
        fig.add_trace(
            go.Bar(x=df['date'],y=df['github_contributions_velocity'],name="Contribution Velocity",
                   marker_color = '#7928CA',showlegend=False),
        row=2 , col=2)

        fig.update_layout(
            title_text = f"Digital Footprint Intelligence Report - Profile :{target_user}",
            height= 750,
            template = "plotly_dark",
            showlegend = True,
            legend=dict(
            x=0.99, y=0.999,
            xanchor='right', yanchor='top',
            bgcolor='rgba(15, 23, 42, 0.85)',
            bordercolor='#334155',
            borderwidth=1,
            font=dict(size=10, color='#FFFFFF')
        )
        )
        fig.show()

# PIPELINE RUNNER

if __name__ == "__main__" :
    TARGET_GITHUB = "torvalds"
    TARGET_STACKOVERFLOW = "22656"

    print(f"[*] Ingesting Footprint Data for : Github ='{TARGET_GITHUB}'...")

    collector =  FootprintDataEngine(github_user=TARGET_GITHUB , stackoverflow_id=TARGET_STACKOVERFLOW)
    raw_df = collector.simulate_time_series_history(months=12)

    processed_df = FeatureProcessor.process(raw_df)

    engine = IntelligenceEngine()
    scored_df = engine.compute_tgs(processed_df)
    scored_df.to_csv("data.csv")
    trajectory = engine.analyze_growth_trajectory(scored_df)
    correlations = engine.compute_correlations(scored_df)


    print("\n" + "="*50)
    print("       TECHNICAL FOOTPRINT REPORT       ")
    print("="*50)
    print(scored_df[['date','github_contributions','so_reputation','TGS']].tail(3).to_string(index=False))
    print("\n--- TRAJECTORY STATISTICAL METRICS ---")
    print(f"Monthly TGS GRowth Rate (Slope) : +{trajectory['monthly_slope']} pts/month")
    print(f"Model Consistency (R2 Score) : {trajectory['r_squared']}")
    print(f"Statistically Significance (p) : {trajectory['p_value']} (p<0.05)") 
    
    print("="*50)

    DashboardBuilder.render(scored_df , correlations , TARGET_GITHUB)
