# ai_agent_parser.py - ENHANCED VERSION WITH ALL 10 FEATURES
"""
AI AGENT QUERY PARSER - ENHANCED
---------------------------------
Enhanced NLP parser supporting 8+ intents with context awareness
"""

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

class QueryParser:
    """Parse natural language queries with context awareness."""
    
    def __init__(self):
        # Core intents patterns
        self.patterns = {
            'who': {
                'patterns': [
                    r'\b(who|which|show|list|display).*(high|critical|elevated|risky|risk)',
                    r'\b(top|highest|most).*(risk|risky)',
                    r'\blist.*(employee|person|people).*risk',
                    r'\bfind.*risk',
                ],
                'description': 'Find high-risk people or teams'
            },
            'why': {
                'patterns': [
                    r'\b(why|explain|reason|what).*(risk|high|critical)',
                    r'\bwhat.*(cause|factor|make|drive).*(risk)',
                    r'\bexplain.*risk',
                    r'\bbreak.*down.*risk',
                ],
                'description': 'Explain reasons behind risk'
            },
            'trend': {
                'patterns': [
                    r'\b(trend|history|over time|pattern|evolution|trajectory)',
                    r'\b(show|display).*(week|month|time|change)',
                    r'\b(how.*changed|track|monitor)',
                    r'\brisk.*over.*time',
                ],
                'description': 'Show risk trends over time'
            },
            'stats': {
                'patterns': [
                    r'\b(how many|count|number of|percentage|stats|statistics)',
                    r'\b(average|mean|total|sum|median)',
                    r'\b(breakdown|distribution|summary)',
                    r'\b(risk.*distribution|overall.*risk)',
                ],
                'description': 'Provide numeric summaries'
            },
            'compare': {
                'patterns': [
                    r'\b(compare|vs|versus|difference|contrast|between)',
                    r'\b(who.*more|who.*higher|who.*worse|better.*than)',
                    r'\bcompare.*team',
                    r'\bdifference.*between',
                ],
                'description': 'Compare entities'
            },
            'feature': {
                'patterns': [
                    r'\b(what|show|explain).*(feature|factor|driver|metric)',
                    r'\b(betweenness|degree|centrality|after.*hours|email)',
                    r'\btop.*feature',
                    r'\bkey.*driver',
                ],
                'description': 'Show key risk drivers'
            },
            'recommend': {
                'patterns': [
                    r'\b(recommend|suggest|advice|help|what.*do|how.*reduce)',
                    r'\b(action|step|intervention|mitigation)',
                    r'\bwhat.*should.*do',
                    r'\bhow.*fix|improve|lower',
                ],
                'description': 'Provide recommendations'
            },
            'what_if': {
                'patterns': [
                    r'\b(what if|simulate|predict|forecast|scenario)',
                    r'\bif.*increase|if.*decrease|if.*change',
                    r'\bwhat.*would.*happen',
                    r'\bimpact.*of',
                ],
                'description': 'Run what-if scenarios'
            },
        }
        
        # Entity patterns
        self.email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        self.number_pattern = r'\b\d+(?:\.\d+)?\b'
        self.team_pattern = r'\b(trading|legal|energy|finance|operations|hr|it|sales)\b'
        
        # Feature keywords
        self.feature_keywords = {
            'betweenness': ['betweenness', 'centrality', 'bridge', 'bottleneck'],
            'degree': ['degree', 'connections', 'contacts', 'network'],
            'after_hours': ['after hours', 'after-hours', 'overtime', 'late', 'weekend'],
            'emails': ['email', 'messages', 'communication', 'volume'],
            'clustering': ['clustering', 'cohesion', 'team', 'group'],
        }
        
        # Time indicators
        self.time_indicators = {
            'current': ['this week', 'current', 'now', 'today', 'latest'],
            'past_week': ['last week', 'previous week', 'week ago'],
            'past_month': ['last month', 'previous month', 'month ago'],
            'custom': r'\b(\d{4}-\d{2}-\d{2})',  # Date pattern
        }
        
        # Context memory
        self.context = {
            'last_entities': [],
            'last_intent': None,
            'last_features': [],
            'last_time_period': 'current',
        }
    
    def parse(self, query: str, update_context: bool = True) -> Dict:
        """
        Parse a user query with context awareness.
        
        Returns:
            dict: Parsed query with intent, entities, confidence, and context
        """
        query_lower = query.lower().strip()
        
        # Detect intent
        intent, confidence = self._detect_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query, query_lower)
        
        # Apply context if needed
        entities = self._apply_context(entities, intent)
        
        # Update context for next query
        if update_context:
            self._update_context(intent, entities)
        
        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
            'original_query': query,
            'context': self.context.copy(),
            'timestamp': datetime.now().isoformat(),
        }
    
    def _detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect the primary intent with confidence score."""
        best_intent = 'unknown'
        best_confidence = 0.0
        
        for intent_name, intent_data in self.patterns.items():
            for pattern in intent_data['patterns']:
                match = re.search(pattern, query)
                if match:
                    # Calculate confidence based on match length and position
                    match_length = len(match.group())
                    query_length = len(query)
                    position_score = 1.0 - (match.start() / query_length)
                    length_score = match_length / query_length
                    confidence = (position_score + length_score) / 2
                    
                    if confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = confidence
        
        # Boost confidence if multiple patterns match
        if best_confidence > 0:
            best_confidence = min(1.0, best_confidence * 1.2)
        
        return best_intent, best_confidence
    
    def _extract_entities(self, query: str, query_lower: str) -> Dict:
        """Extract all entities from the query."""
        entities = {}
        
        # Extract emails
        emails = re.findall(self.email_pattern, query.lower())
        if emails:
            entities['emails'] = list(set(emails))
        
        # Extract numbers
        numbers = re.findall(self.number_pattern, query)
        if numbers:
            entities['numbers'] = [float(n) for n in numbers]
        
        # Extract team names
        teams = re.findall(self.team_pattern, query_lower)
        if teams:
            entities['teams'] = list(set(teams))
        
        # Extract risk bands
        risk_bands = []
        for band in ['critical', 'high', 'elevated', 'medium', 'low']:
            if band in query_lower:
                risk_bands.append(band.title())
        if risk_bands:
            entities['risk_bands'] = risk_bands
        
        # Extract feature names
        features = []
        for feature_key, keywords in self.feature_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    features.append(feature_key)
                    break
        if features:
            entities['features'] = list(set(features))
        
        # Extract time period
        time_period = self._extract_time_period(query_lower)
        if time_period:
            entities['time_period'] = time_period
        
        # Extract percentage changes for what-if
        if 'what_if' in query_lower or 'simulate' in query_lower:
            percent_pattern = r'(\d+)\s*%'
            percents = re.findall(percent_pattern, query)
            if percents:
                entities['change_percent'] = [int(p) for p in percents]
            
            # Extract increase/decrease direction
            if 'increase' in query_lower or 'up' in query_lower:
                entities['change_direction'] = 'increase'
            elif 'decrease' in query_lower or 'reduce' in query_lower or 'down' in query_lower:
                entities['change_direction'] = 'decrease'
        
        return entities
    
    def _extract_time_period(self, query: str) -> Optional[str]:
        """Extract time period from query."""
        for period_name, indicators in self.time_indicators.items():
            if period_name == 'custom':
                # Check for date pattern
                date_match = re.search(indicators, query)
                if date_match:
                    return date_match.group()
            else:
                # Check for keyword indicators
                for indicator in indicators:
                    if indicator in query:
                        return period_name
        
        return None
    
    def _apply_context(self, entities: Dict, intent: str) -> Dict:
        """Apply context from previous queries if needed."""
        # If no emails specified but context has them, use context
        if 'emails' not in entities and self.context['last_entities']:
            # Check if query references previous context
            context_references = ['same', 'them', 'that', 'those', 'again', 'also']
            if any(ref in entities.get('original_query', '').lower() for ref in context_references):
                entities['emails'] = self.context['last_entities']
        
        # If comparing but only one entity, use last entity as second
        if intent == 'compare' and 'emails' in entities:
            if len(entities['emails']) == 1 and self.context['last_entities']:
                entities['emails'].extend(self.context['last_entities'][:1])
        
        # Apply time period from context if not specified
        if 'time_period' not in entities and self.context['last_time_period']:
            entities['time_period'] = self.context['last_time_period']
        
        return entities
    
    def _update_context(self, intent: str, entities: Dict):
        """Update context for next query."""
        self.context['last_intent'] = intent
        
        if 'emails' in entities:
            self.context['last_entities'] = entities['emails']
        
        if 'features' in entities:
            self.context['last_features'] = entities['features']
        
        if 'time_period' in entities:
            self.context['last_time_period'] = entities['time_period']
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of intent."""
        return self.patterns.get(intent, {}).get('description', 'Unknown intent')
    
    def reset_context(self):
        """Reset conversation context."""
        self.context = {
            'last_entities': [],
            'last_intent': None,
            'last_features': [],
            'last_time_period': 'current',
        }


if __name__ == "__main__":
    parser = QueryParser()
    
    # Test queries
    test_queries = [
        "Who is high risk this week?",
        "Why is christy.wire@enron.com high risk?",
        "Compare them with john.doe@enron.com",  # Uses context
        "Show trend over last month",
        "What if we reduce after-hours emails by 30%?",
        "How many people are critical in trading team?",
        "What are the top risk drivers?",
        "Recommend actions for high risk employees",
    ]
    
    print("ENHANCED PARSER TEST WITH CONTEXT")
    print("=" * 70)
    
    for query in test_queries:
        result = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"  Entities: {result['entities']}")
        if result['context']['last_entities']:
            print(f"  Context: Using previous entities: {result['context']['last_entities']}")