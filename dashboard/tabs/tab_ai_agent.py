# tab_ai_agent.py - FIXED VERSION
"""
AI Agent Chat Tab - Fixed parser scope error
"""
import streamlit as st
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import json

# Add ai_agent folder to path
AI_AGENT_DIR = Path(__file__).parent.parent / "ai_agent"
if str(AI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AI_AGENT_DIR))

def render(data: dict, filters: dict):
    """Render enhanced AI Agent chat tab with all features."""
    
    st.subheader("AI Risk Agent - Enhanced Intelligence")
    st.caption("Natural language interface with model insights, simulations, and automated reporting")
    
    # Load AI Agent components
    @st.cache_resource
    def load_ai_agent():
        try:
            from ai_agent_parser import QueryParser
            from ai_agent_responder import ResponseGenerator
            return QueryParser(), ResponseGenerator(), None
        except Exception as e:
            return None, None, str(e)
    
    parser, responder, error = load_ai_agent()
    
    if error:
        st.error(f"AI Agent initialization error: {error}")
        return
    
    # Store parser in session state for access in other functions
    st.session_state.parser = parser
    st.session_state.responder = responder
    
    # Initialize session state
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
        st.session_state.ai_context = {
            'session_start': datetime.now().isoformat(),
            'query_count': 0,
            'last_report_date': None,
            'conversation_memory': []
        }
    
    # Sidebar controls with enhanced features
    with st.sidebar:
        st.markdown("---")
        st.subheader("AI Agent Controls")
        
        # Feature toggles
        st.markdown("**Features:**")
        col1, col2 = st.columns(2)
        
        with col1:
            show_confidence = st.checkbox("Show Confidence", value=True, key="ai_show_conf")
            show_evidence = st.checkbox("Show Evidence", value=True, key="ai_show_evidence")
            enable_memory = st.checkbox("Context Memory", value=True, key="ai_enable_memory")
        
        with col2:
            show_charts = st.checkbox("Visualizations", value=True, key="ai_show_charts")
            auto_report = st.checkbox("Auto Reports", value=False, key="ai_auto_report")
            enable_simulation = st.checkbox("Simulations", value=True, key="ai_enable_sim")
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("Quick Actions")
        
        quick_actions = {
            "Weekly Report": "Generate weekly risk report",
            "Top Risks": "Who are the top 10 high-risk employees?",
            "Team Analysis": "Show risk statistics by team",
            "Critical Alert": "Show all critical risk employees",
            "Trend Analysis": "Show risk trends for the past month",
            "Feature Impact": "What are the top risk drivers?",
            "Simulate": "What if we reduce after-hours emails by 50%?",
            "Recommendations": "What actions should we take for high-risk employees?"
        }
        
        for label, query in quick_actions.items():
            if st.button(label, key=f"ai_quick_{label}", use_container_width=True):
                # Process the query
                parsed = parser.parse(query)
                st.session_state.ai_messages.append({
                    "role": "user",
                    "content": {"text": query}
                })
                
                # Generate response
                response = responder.generate_response(parsed)
                st.session_state.ai_messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.ai_context['query_count'] += 1
                st.rerun()
        
        st.markdown("---")
        
        # Conversation management
        st.subheader("Conversation")
        
        if st.button("Clear Chat", key="ai_clear", use_container_width=True):
            st.session_state.ai_messages = []
            parser.reset_context()
            st.success("Chat cleared!")
            st.rerun()
        
        if st.button("Export Chat", key="ai_export", use_container_width=True):
            export_conversation()
        
        # Memory status
        if enable_memory:
            st.caption(f"Context: {st.session_state.ai_context['query_count']} queries")
            if parser.context['last_entities']:
                st.caption(f"Tracking: {', '.join(parser.context['last_entities'][:2])}")
    
    # Main chat area with tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Insights", "Reports"])
    
    with tab1:
        render_chat(parser, responder, show_charts, show_confidence, show_evidence)
    
    with tab2:
        render_insights(data)
    
    with tab3:
        render_reports(responder, auto_report)
    
    # Footer with capabilities
    st.markdown("---")
    with st.expander("AI Agent Capabilities", expanded=False):
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("**Understanding**")
            st.caption("- Natural language\n- Context awareness\n- Entity recognition\n- Intent detection")
        
        with cols[1]:
            st.markdown("**Analytics**")
            st.caption("- SHAP explanations\n- Risk predictions\n- Trend analysis\n- Team statistics")
        
        with cols[2]:
            st.markdown("**Simulations**")
            st.caption("- What-if scenarios\n- Impact prediction\n- Intervention testing\n- Risk modeling")
        
        with cols[3]:
            st.markdown("**Automation**")
            st.caption("- Weekly reports\n- Auto-summaries\n- Export functions\n- Alert generation")


def render_chat(parser, responder, show_charts, show_confidence, show_evidence):
    """Render the main chat interface."""
    
    # Initialize with welcome message if empty
    if not st.session_state.ai_messages:
        welcome_response = responder.generate_weekly_report()
        st.session_state.ai_messages.append({
            "role": "assistant",
            "content": {
                "text": f"""**Welcome to Enhanced AI Risk Agent!**

I have advanced capabilities including:
- Natural language understanding with context memory
- Model-driven insights with SHAP explanations
- What-if scenario simulations
- Automated weekly reporting
- Evidence-based recommendations

**Current Status:**
{welcome_response}

What would you like to know?""",
                "figure": None,
                "data": None
            }
        })
    
    # Display conversation history
    for idx, message in enumerate(st.session_state.ai_messages):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            if isinstance(content, dict):
                # Display text
                text = content.get("text", "")
                
                # Add confidence if available
                if show_confidence and content.get("metadata"):
                    confidence = content["metadata"].get("confidence", 0)
                    if confidence > 0:
                        text = f"[Confidence: {confidence:.0%}]\n\n{text}"
                
                st.markdown(text)
                
                # Display visualization if available
                if show_charts and content.get("figure"):
                    chart_key = f"chart_{idx}_{int(time.time()*1000000)}"
                    st.plotly_chart(content["figure"], use_container_width=True, key=chart_key)
                
                # Display data table if available
                if show_evidence and content.get("data"):
                    if isinstance(content["data"], list) and len(content["data"]) > 0:
                        with st.expander("View Data", expanded=False):
                            df = pd.DataFrame(content["data"])
                            st.dataframe(df, use_container_width=True)
                    elif isinstance(content["data"], dict):
                        with st.expander("View Details", expanded=False):
                            st.json(content["data"])
            else:
                st.markdown(content)
    
    # Chat input with enhanced processing
    if prompt := st.chat_input("Ask about employee risk..."):
        # Add user message
        st.session_state.ai_messages.append({
            "role": "user",
            "content": {"text": prompt}
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with progress
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query..."):
                try:
                    # Parse with context
                    parsed = parser.parse(prompt, update_context=True)
                    
                    # Show parsing results if in debug mode
                    if st.session_state.get("debug_mode", False):
                        st.caption(f"Intent: {parsed['intent']} | Entities: {parsed['entities']}")
                    
                    # Generate response
                    with st.spinner("Generating response..."):
                        response = responder.generate_response(parsed)
                    
                    # Display response
                    text = response.get("text", "No response generated")
                    
                    # Add confidence if shown
                    if show_confidence and response.get("metadata"):
                        confidence = response["metadata"].get("confidence", 0)
                        if confidence > 0:
                            text = f"[Confidence: {confidence:.0%}]\n\n{text}"
                    
                    st.markdown(text)
                    
                    # Display visualization
                    if show_charts and response.get("figure"):
                        chart_key = f"chart_new_{len(st.session_state.ai_messages)}_{int(time.time()*1000000)}"
                        st.plotly_chart(response["figure"], use_container_width=True, key=chart_key)
                    
                    # Display data
                    if show_evidence and response.get("data"):
                        if isinstance(response["data"], list) and len(response["data"]) > 0:
                            with st.expander("View Data", expanded=False):
                                df = pd.DataFrame(response["data"])
                                st.dataframe(df, use_container_width=True)
                    
                    # Store response
                    st.session_state.ai_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Update context
                    st.session_state.ai_context['query_count'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.ai_messages.append({
                        "role": "assistant",
                        "content": {"text": error_msg, "figure": None, "data": None}
                    })


def render_insights(data: dict):
    """Render the insights tab with key findings."""
    
    st.subheader("AI-Generated Insights")
    
    # Get parser from session state
    parser = st.session_state.get('parser', None)
    
    # Generate insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", st.session_state.ai_context['query_count'])
    
    with col2:
        session_duration = (datetime.now() - datetime.fromisoformat(st.session_state.ai_context['session_start'])).total_seconds() / 60
        st.metric("Session Duration", f"{session_duration:.0f} min")
    
    with col3:
        # Calculate unique entities safely
        unique_entities = 0
        if parser:
            entities_list = []
            for msg in st.session_state.ai_messages:
                if msg['role'] == 'user' and parser.context.get('last_entities'):
                    entities_list.extend(parser.context.get('last_entities', []))
            unique_entities = len(set(entities_list))
        st.metric("Entities Analyzed", unique_entities)
    
    st.markdown("---")
    
    # Key findings from conversation
    st.subheader("Key Findings from Conversation")
    
    findings = []
    for msg in st.session_state.ai_messages:
        if msg['role'] == 'assistant' and isinstance(msg['content'], dict):
            metadata = msg['content'].get('metadata', {})
            if metadata.get('intent') == 'who':
                data = msg['content'].get('data')
                if data and isinstance(data, list):
                    findings.append(f"- Identified {len(data)} high-risk employees")
            elif metadata.get('intent') == 'why':
                findings.append(f"- Analyzed risk factors with SHAP explanations")
            elif metadata.get('intent') == 'what_if':
                findings.append(f"- Simulated intervention scenarios")
    
    if findings:
        for finding in findings[:5]:
            st.write(finding)
    else:
        st.info("No significant findings yet. Start asking questions!")
    
    st.markdown("---")
    
    # Recommendations summary
    st.subheader("Aggregated Recommendations")
    
    recommendations = set()
    for msg in st.session_state.ai_messages:
        if msg['role'] == 'assistant' and isinstance(msg['content'], dict):
            text = msg['content'].get('text', '')
            if 'Recommendation' in text or 'Action' in text:
                # Extract recommendation lines
                lines = text.split('\n')
                for line in lines:
                    if any(keyword in line for keyword in ['Set', 'Implement', 'Review', 'Monitor', 'Schedule']):
                        cleaned_line = line.strip('- ').strip()
                        if cleaned_line:
                            recommendations.add(cleaned_line)
    
    if recommendations:
        for i, rec in enumerate(list(recommendations)[:5], 1):
            st.write(f"{i}. {rec}")
    else:
        st.info("No recommendations generated yet.")


def render_reports(responder, auto_report):
    """Render the reports tab."""
    
    st.subheader("Automated Reports")
    
    # Report generation controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Weekly Report", use_container_width=True):
            with st.spinner("Generating report..."):
                report = responder.generate_weekly_report()
                st.session_state.last_report = report
                st.session_state.ai_context['last_report_date'] = datetime.now().isoformat()
    
    with col2:
        if st.button("Generate Executive Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = generate_executive_summary(responder)
                st.session_state.last_summary = summary
    
    st.markdown("---")
    
    # Display last report
    if hasattr(st.session_state, 'last_report'):
        st.markdown("### Last Generated Report")
        st.markdown(st.session_state.last_report)
        
        # Download button
        st.download_button(
            label="Download Report (Markdown)",
            data=st.session_state.last_report,
            file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
    
    # Display last summary
    if hasattr(st.session_state, 'last_summary'):
        st.markdown("---")
        st.markdown("### Executive Summary")
        st.markdown(st.session_state.last_summary)
    
    # Auto-report status
    if auto_report:
        st.markdown("---")
        st.info("Auto-reporting is enabled. Reports will be generated weekly.")
        if st.session_state.ai_context['last_report_date']:
            last_date = datetime.fromisoformat(st.session_state.ai_context['last_report_date'])
            st.caption(f"Last report: {last_date.strftime('%Y-%m-%d %H:%M')}")


def export_conversation():
    """Export conversation to file."""
    
    export_data = {
        'session_info': st.session_state.ai_context,
        'messages': []
    }
    
    for msg in st.session_state.ai_messages:
        export_msg = {
            'role': msg['role'],
            'content': msg['content'].get('text', '') if isinstance(msg['content'], dict) else msg['content']
        }
        export_data['messages'].append(export_msg)
    
    # Convert to markdown
    markdown = f"""# AI Agent Conversation Export
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Session Duration: {st.session_state.ai_context['query_count']} queries

---

"""
    
    for msg in export_data['messages']:
        role = "User" if msg['role'] == 'user' else "AI Agent"
        markdown += f"**{role}:** {msg['content']}\n\n---\n\n"
    
    # Provide download
    st.download_button(
        label="Download Conversation",
        data=markdown,
        file_name=f"ai_conversation_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )
    
    st.success("Conversation exported!")


def generate_executive_summary(responder):
    """Generate executive summary from conversation."""
    
    summary_parts = [
        "# Executive Summary",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Key Metrics"
    ]
    
    # Analyze messages for key metrics
    high_risk_count = 0
    critical_count = 0
    
    for msg in st.session_state.ai_messages:
        if msg['role'] == 'assistant' and isinstance(msg['content'], dict):
            data = msg['content'].get('data')
            if data:
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            risk_score = item.get('risk_score', 0)
                            if risk_score > 0.75:
                                critical_count += 1
                            elif risk_score > 0.5:
                                high_risk_count += 1
    
    summary_parts.extend([
        f"- High Risk Employees Identified: {high_risk_count}",
        f"- Critical Risk Employees: {critical_count}",
        f"- Total Queries Processed: {st.session_state.ai_context['query_count']}",
        "",
        "## Priority Actions",
        "1. Address critical risk employees immediately",
        "2. Implement recommended interventions",
        "3. Monitor rising risk trends",
        "4. Review team workload distribution",
        "",
        "## Next Steps",
        "- Schedule intervention meetings with critical risk employees",
        "- Deploy work-life balance initiatives",
        "- Continue monitoring with weekly reports"
    ])
    
    return '\n'.join(summary_parts)