import csv
import random
from datetime import datetime, timedelta
import math

def generate_common_data(num_rows=200):
    countries = ["United States", "China", "India", "Germany", "Japan", "Brazil", "United Kingdom", "France", "Italy", "Canada", 
                 "Australia", "South Korea", "Spain", "Mexico", "Netherlands", "Sweden", "Switzerland", "Singapore", "Russia", "Indonesia"]
    cities = {country: [f"{country}_City_{i}" for i in range(1, 8)] for country in countries}
    companies = ["TechCorp", "EcoSolutions", "GlobalFinance", "HealthInnovate", "EnergyFuture", "SmartRetail", "DataDynamics", 
                 "GreenMobility", "AgroTech", "SpaceVentures", "AIFirst", "CyberSecure", "BioGen", "QuantumLeap", "RoboWorks", 
                 "NanoTech", "CloudPioneer", "FintechFront", "EduTech", "CleanEnergy"]
    industries = ["Information Technology", "Energy", "Finance", "Healthcare", "Retail", "Manufacturing", "Agriculture", "Aerospace", 
                  "Telecommunications", "Education", "Automotive", "Entertainment", "Biotechnology", "Real Estate", "Transportation"]
    
    common_data = []
    used_combinations = set()

    for i in range(num_rows):
        while True:
            country = random.choice(countries)
            company = random.choice(companies)
            industry = random.choice(industries)
            year = random.randint(2020, 2024)
            
            combination = (country, company, industry, year)
            if combination not in used_combinations:
                used_combinations.add(combination)
                break

        industry_ai_readiness = {
            "Information Technology": 0.9, "Energy": 0.6, "Finance": 0.8, "Healthcare": 0.7, "Retail": 0.7,
            "Manufacturing": 0.6, "Agriculture": 0.5, "Aerospace": 0.8, "Telecommunications": 0.85,
            "Education": 0.65, "Automotive": 0.75, "Entertainment": 0.7, "Biotechnology": 0.8,
            "Real Estate": 0.55, "Transportation": 0.65
        }
        
        ai_readiness = industry_ai_readiness[industry] * (1 + 0.1 * (year - 2020)) * random.uniform(0.9, 1.1)
        
        common_data.append({
            "id": i + 1,
            "country": country,
            "city": random.choice(cities[country]),
            "company": company,
            "industry": industry,
            "year": year,
            "ai_readiness": min(ai_readiness, 1.0)
        })
    return common_data

def generate_generative_ai_business_models(common_data):
    ai_applications = ["Product Design", "Content Creation", "Process Optimization", "Customer Service", "Predictive Maintenance", 
                       "Personalization", "Fraud Detection", "Supply Chain Optimization", "Drug Discovery", "Autonomous Systems"]
    disruption_levels = ["Low", "Medium", "High", "Revolutionary"]
    esg_impact_areas = ["Carbon Footprint Reduction", "Waste Management", "Energy Efficiency", "Social Inclusion", 
                        "Ethical Supply Chain", "Transparency", "Water Conservation", "Biodiversity Protection", 
                        "Employee Well-being", "Community Development"]
    
    all_industries = set(entry["industry"] for entry in common_data)
    
    data = []
    for entry in common_data:
        ai_adoption = entry["ai_readiness"] * random.uniform(0.8, 1.2) * 100
        disruption_level = random.choices(disruption_levels, weights=[0.3, 0.4, 0.2, 0.1])[0]
        esg_score = random.uniform(50, 100) * (0.8 + 0.4 * entry["ai_readiness"])
        
        base_revenue_growth = random.uniform(0, 30)
        disruption_multiplier = {"Low": 1, "Medium": 1.2, "High": 1.5, "Revolutionary": 2}
        revenue_growth = base_revenue_growth * disruption_multiplier[disruption_level] * (1 + ai_adoption / 200)
        
        base_cost_reduction = random.uniform(5, 20)
        industry_efficiency = {industry: random.uniform(0.9, 1.4) for industry in all_industries}
        cost_reduction = base_cost_reduction * industry_efficiency[entry["industry"]] * (1 + ai_adoption / 150)
        
        row = {
            "id": entry["id"],
            "company": entry["company"],
            "country": entry["country"],
            "industry": entry["industry"],
            "year": entry["year"],
            "ai_adoption_percentage": round(ai_adoption, 2),
            "primary_ai_application": random.choice(ai_applications),
            "disruption_level": disruption_level,
            "revenue_growth": round(revenue_growth, 2),
            "cost_reduction": round(cost_reduction, 2),
            "esg_score": round(esg_score, 2),
            "primary_esg_impact": random.choice(esg_impact_areas),
            "sustainable_growth_index": round((ai_adoption * 0.6 + esg_score * 0.4) / 100, 2),
            "innovation_index": round(min(100, random.uniform(60, 100) * (1 + ai_adoption / 200)), 2),
            "employee_satisfaction": round(min(100, random.uniform(60, 90) + (esg_score * 0.2)), 2),
            "market_share_change": round(revenue_growth * 0.3 * random.uniform(0.8, 1.2), 2)
        }
        data.append(row)
    return data
def generate_ai_impact_on_traditional_industries(common_data):
    traditional_processes = ["Manufacturing", "Customer Support", "Logistics", "Marketing", "R&D", "Human Resources", 
                             "Quality Control", "Financial Planning", "Inventory Management", "Product Development"]
    ai_technologies = ["Natural Language Processing", "Computer Vision", "Predictive Analytics", "Robotics", 
                       "Generative Models", "Reinforcement Learning", "Expert Systems", "Speech Recognition", 
                       "Autonomous Vehicles", "Quantum Machine Learning"]
    
    all_industries = set(entry["industry"] for entry in common_data)
    
    # Define industry-specific factors
    industry_automation_factor = {
        "Information Technology": 1.5, "Energy": 1.2, "Finance": 1.4, "Healthcare": 1.3, "Retail": 1.3,
        "Manufacturing": 1.4, "Agriculture": 1.1, "Aerospace": 1.5, "Telecommunications": 1.4,
        "Education": 1.0, "Automotive": 1.3, "Entertainment": 1.1, "Biotechnology": 1.2,
        "Real Estate": 1.0, "Transportation": 1.2
    }
    
    # Define technology impact factors
    tech_impact_factor = {
        "Natural Language Processing": 1.3, "Computer Vision": 1.4, "Predictive Analytics": 1.2,
        "Robotics": 1.5, "Generative Models": 1.1, "Reinforcement Learning": 1.2,
        "Expert Systems": 1.0, "Speech Recognition": 1.1, "Autonomous Vehicles": 1.4,
        "Quantum Machine Learning": 1.6
    }
    
    data = []
    for entry in common_data:
        ai_investment = entry["ai_readiness"] * random.uniform(0.9, 1.1) * 100
        
        # Select AI technology with some bias towards industry
        if entry["industry"] in ["Information Technology", "Finance", "Telecommunications"]:
            ai_tech_weights = [1.5 if tech in ["Natural Language Processing", "Predictive Analytics", "Generative Models"] else 1 for tech in ai_technologies]
        elif entry["industry"] in ["Manufacturing", "Automotive", "Aerospace"]:
            ai_tech_weights = [1.5 if tech in ["Robotics", "Computer Vision", "Autonomous Vehicles"] else 1 for tech in ai_technologies]
        else:
            ai_tech_weights = [1] * len(ai_technologies)
        
        ai_technology = random.choices(ai_technologies, weights=ai_tech_weights)[0]
        
        # Calculate process efficiency based on AI investment, industry, and technology
        base_efficiency = ai_investment * 0.5
        industry_factor = industry_automation_factor.get(entry["industry"], 1.0)
        tech_factor = tech_impact_factor[ai_technology]
        process_efficiency = base_efficiency * industry_factor * tech_factor * random.uniform(0.9, 1.1)
        
        # Calculate jobs automated based on efficiency and industry
        base_jobs_automated = process_efficiency * 0.2
        jobs_automated = base_jobs_automated * industry_automation_factor.get(entry["industry"], 1.0) * random.uniform(0.9, 1.1)
        
        # New jobs created as a function of automated jobs and AI readiness
        new_jobs_created = jobs_automated * (0.5 + 0.5 * entry["ai_readiness"]) * random.uniform(0.9, 1.1)
        
        # Cost savings as a function of process efficiency and jobs automated
        cost_savings = (process_efficiency * 0.3 + jobs_automated * 0.7) * random.uniform(0.9, 1.1)
        
        # Product quality improvement based on AI investment and technology
        product_quality_improvement = ai_investment * 0.2 * tech_impact_factor[ai_technology] * random.uniform(0.9, 1.1)
        
        # Time to market reduction based on process efficiency and industry
        time_to_market_reduction = process_efficiency * 0.4 * industry_automation_factor.get(entry["industry"], 1.0) * random.uniform(0.9, 1.1)
        
        row = {
            "id": entry["id"],
            "company": entry["company"],
            "industry": entry["industry"],
            "year": entry["year"],
            "ai_investment_percentage": round(ai_investment, 2),
            "traditional_process_impacted": random.choice(traditional_processes),
            "ai_technology_used": ai_technology,
            "process_efficiency_improvement": round(process_efficiency, 2),
            "cost_savings": round(cost_savings, 2),
            "jobs_automated": round(jobs_automated, 2),
            "new_jobs_created": round(new_jobs_created, 2),
            "product_quality_improvement": round(product_quality_improvement, 2),
            "time_to_market_reduction": round(time_to_market_reduction, 2)
        }
        data.append(row)
    return data

def generate_ai_esg_alignment(common_data):
    esg_initiatives = ["Carbon Neutrality", "Circular Economy", "Diversity & Inclusion", "Ethical AI", "Sustainable Supply Chain", 
                       "Community Engagement", "Renewable Energy Adoption", "Water Conservation", "Fair Labor Practices", 
                       "Biodiversity Protection"]
    ai_contributions = ["Emissions Tracking", "Waste Reduction Optimization", "Bias Detection", "Ethical Decision Support", 
                        "Supplier Risk Assessment", "Social Impact Prediction", "Energy Efficiency Modeling", 
                        "Ecosystem Monitoring", "Fair Pay Analysis", "Sustainable Resource Allocation"]
    
    # Define industry-specific ESG factors
    industry_esg_factor = {
        "Information Technology": 1.2, "Energy": 1.5, "Finance": 1.1, "Healthcare": 1.3, "Retail": 1.0,
        "Manufacturing": 1.4, "Agriculture": 1.3, "Aerospace": 1.2, "Telecommunications": 1.1,
        "Education": 1.0, "Automotive": 1.3, "Entertainment": 0.9, "Biotechnology": 1.2,
        "Real Estate": 1.1, "Transportation": 1.4
    }
    
    # Define AI contribution impact factors
    ai_contribution_impact = {
        "Emissions Tracking": 1.3, "Waste Reduction Optimization": 1.4, "Bias Detection": 1.2,
        "Ethical Decision Support": 1.1, "Supplier Risk Assessment": 1.3, "Social Impact Prediction": 1.2,
        "Energy Efficiency Modeling": 1.5, "Ecosystem Monitoring": 1.4, "Fair Pay Analysis": 1.1,
        "Sustainable Resource Allocation": 1.3
    }
    
    data = []
    for entry in common_data:
        # Calculate AI ESG investment based on industry and AI readiness
        base_ai_esg_investment = entry["ai_readiness"] * random.uniform(0.8, 1.2) * 100
        industry_factor = industry_esg_factor.get(entry["industry"], 1.0)
        ai_esg_investment = base_ai_esg_investment * industry_factor
        
        # Select primary ESG initiative with some bias towards industry
        if entry["industry"] in ["Energy", "Manufacturing", "Agriculture"]:
            esg_weights = [1.5 if init in ["Carbon Neutrality", "Circular Economy", "Renewable Energy Adoption"] else 1 for init in esg_initiatives]
        elif entry["industry"] in ["Information Technology", "Finance", "Healthcare"]:
            esg_weights = [1.5 if init in ["Ethical AI", "Diversity & Inclusion", "Fair Labor Practices"] else 1 for init in esg_initiatives]
        else:
            esg_weights = [1] * len(esg_initiatives)
        
        primary_esg_initiative = random.choices(esg_initiatives, weights=esg_weights)[0]
        
        # Select AI contribution based on primary ESG initiative
        relevant_contributions = {
            "Carbon Neutrality": ["Emissions Tracking", "Energy Efficiency Modeling"],
            "Circular Economy": ["Waste Reduction Optimization", "Sustainable Resource Allocation"],
            "Diversity & Inclusion": ["Bias Detection", "Fair Pay Analysis"],
            "Ethical AI": ["Ethical Decision Support", "Bias Detection"],
            "Sustainable Supply Chain": ["Supplier Risk Assessment", "Sustainable Resource Allocation"],
            "Community Engagement": ["Social Impact Prediction", "Ecosystem Monitoring"],
            "Renewable Energy Adoption": ["Energy Efficiency Modeling", "Emissions Tracking"],
            "Water Conservation": ["Sustainable Resource Allocation", "Ecosystem Monitoring"],
            "Fair Labor Practices": ["Fair Pay Analysis", "Ethical Decision Support"],
            "Biodiversity Protection": ["Ecosystem Monitoring", "Social Impact Prediction"]
        }
        ai_contribution = random.choice(relevant_contributions[primary_esg_initiative])
        
        # Calculate ESG performance score
        base_esg_performance = ai_esg_investment * 0.5
        contribution_factor = ai_contribution_impact[ai_contribution]
        esg_performance = base_esg_performance * contribution_factor * random.uniform(0.9, 1.1)
        
        # Calculate carbon footprint reduction
        base_carbon_reduction = esg_performance * 0.3
        carbon_footprint_reduction = base_carbon_reduction * industry_factor * random.uniform(0.9, 1.1)
        
        # Calculate resource efficiency improvement
        base_efficiency = ai_esg_investment * 0.2
        resource_efficiency_improvement = base_efficiency * contribution_factor * random.uniform(0.9, 1.1)
        
        # Calculate stakeholder trust index
        base_trust = 60 + (esg_performance * 0.3)
        stakeholder_trust_index = min(100, base_trust * random.uniform(0.95, 1.05))
        
        # Calculate regulatory compliance score
        base_compliance = 75 + (ai_esg_investment * 0.1)
        regulatory_compliance_score = min(100, base_compliance * industry_factor * random.uniform(0.97, 1.03))
        
        # Calculate social impact score
        base_social_impact = 50 + (esg_performance * 0.4)
        social_impact_score = min(100, base_social_impact * contribution_factor * random.uniform(0.95, 1.05))
        
        row = {
            "id": entry["id"],
            "company": entry["company"],
            "industry": entry["industry"],
            "year": entry["year"],
            "ai_esg_investment_percentage": round(ai_esg_investment, 2),
            "primary_esg_initiative": primary_esg_initiative,
            "ai_contribution": ai_contribution,
            "esg_performance_score": round(esg_performance, 2),
            "carbon_footprint_reduction": round(carbon_footprint_reduction, 2),
            "resource_efficiency_improvement": round(resource_efficiency_improvement, 2),
            "stakeholder_trust_index": round(stakeholder_trust_index, 2),
            "regulatory_compliance_score": round(regulatory_compliance_score, 2),
            "social_impact_score": round(social_impact_score, 2)
        }
        data.append(row)
    return data


def write_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# Generate common data
common_data = generate_common_data(200)

# Generate and write sample data for each dataset
datasets = [
    (generate_generative_ai_business_models, 'generative_ai_business_models.csv'),
    (generate_ai_impact_on_traditional_industries, 'ai_impact_on_traditional_industries.csv'),
    (generate_ai_esg_alignment, 'ai_esg_alignment.csv')
]

for generator_func, filename in datasets:
    data = generator_func(common_data)
    write_csv(data, filename)

print("Enhanced Generative AI business models datasets with increased uniqueness have been generated and saved as CSV files.")
