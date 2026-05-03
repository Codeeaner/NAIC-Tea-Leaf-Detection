"""Report generation service for detection results."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from sqlalchemy.orm import Session
from app.models.detection import DetectionSession, DetectionResult


class ReportService:
    """Service for generating reports from detection results."""
    
    def __init__(self, db: Session):
        """Initialize report service."""
        self.db = db
        self.reports_dir = "reports"
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_pdf_report(self, session_id: int) -> Optional[str]:
        """
        Generate a PDF report for a detection session.
        
        Args:
            session_id: ID of the detection session
            
        Returns:
            Path to the generated PDF file or None if failed
        """
        
        try:
            # Get session data
            session = self.db.query(DetectionSession).filter(
                DetectionSession.id == session_id
            ).first()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tea_leaf_report_session_{session_id}_{timestamp}.pdf"
            filepath = os.path.join(self.reports_dir, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            elements = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.darkgreen
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Title
            elements.append(Paragraph("Tea Leaf Health Detection Report", title_style))
            elements.append(Spacer(1, 20))
            
            # Session information
            elements.append(Paragraph("Session Information", heading_style))
            session_info = [
                ["Session Name:", session.name],
                ["Session ID:", str(session.id)],
                ["Created:", session.created_at.strftime("%Y-%m-%d %H:%M:%S")],
                ["Status:", session.status.title()],
                ["Total Images:", str(session.total_images)],
                ["Processed Images:", str(session.processed_images)]
            ]
            
            session_table = Table(session_info, colWidths=[2*inch, 3*inch])
            session_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ]))
            
            elements.append(session_table)
            elements.append(Spacer(1, 20))
            
            # Calculate statistics
            stats = self._calculate_session_statistics(session)
            
            # Summary statistics
            elements.append(Paragraph("Detection Summary", heading_style))
            summary_data = [
                ["Total Leaves Detected:", str(stats['total_leaves'])],
                ["Healthy Leaves:", f"{stats['total_healthy']} ({stats['healthy_percentage']:.1f}%)"],
                ["Unhealthy Leaves:", f"{stats['total_unhealthy']} ({stats['unhealthy_percentage']:.1f}%)"],
                ["Average Health Score:", f"{stats['average_health_percentage']:.1f}%"],
                ["Average Processing Time:", f"{stats['average_processing_time']:.2f} seconds"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 2.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.lightyellow),
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Generate charts
            chart_paths = self._generate_charts(session, stats)
            
            # Add charts to PDF
            if chart_paths:
                elements.append(Paragraph("Visual Analysis", heading_style))
                
                for chart_path in chart_paths:
                    if os.path.exists(chart_path):
                        # Resize image to fit page
                        img = Image(chart_path, width=5*inch, height=3*inch)
                        elements.append(img)
                        elements.append(Spacer(1, 10))
            
            # Detailed results table
            if len(session.results) > 0:
                elements.append(Paragraph("Detailed Results", heading_style))
                
                # Create detailed results table
                results_data = [["Image Name", "Healthy", "Unhealthy", "Total", "Health %", "Status"]]
                
                for result in session.results[:20]:  # Limit to first 20 results
                    results_data.append([
                        result.image_name[:30] + "..." if len(result.image_name) > 30 else result.image_name,
                        str(result.healthy_count),
                        str(result.unhealthy_count),
                        str(result.total_count),
                        f"{result.health_percentage:.1f}%",
                        result.status.title()
                    ])
                
                results_table = Table(results_data, colWidths=[2*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(results_table)
            
            # Build PDF
            doc.build(elements)
            
            # Clean up chart files
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    os.remove(chart_path)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            return None
    
    def generate_csv_report(self, session_id: int) -> Optional[str]:
        """
        Generate a CSV report for a detection session.
        
        Args:
            session_id: ID of the detection session
            
        Returns:
            Path to the generated CSV file or None if failed
        """
        
        try:
            # Get session data
            session = self.db.query(DetectionSession).filter(
                DetectionSession.id == session_id
            ).first()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Prepare data for CSV
            data = []
            for result in session.results:
                data.append({
                    'session_id': session.id,
                    'session_name': session.name,
                    'image_name': result.image_name,
                    'image_path': result.image_path,
                    'healthy_count': result.healthy_count,
                    'unhealthy_count': result.unhealthy_count,
                    'total_count': result.total_count,
                    'health_percentage': result.health_percentage,
                    'confidence_threshold': result.confidence_threshold,
                    'processing_time': result.processing_time,
                    'status': result.status,
                    'error_message': result.error_message,
                    'created_at': result.created_at,
                    'annotated_image_path': result.annotated_image_path
                })
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tea_leaf_data_session_{session_id}_{timestamp}.csv"
            filepath = os.path.join(self.reports_dir, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating CSV report: {e}")
            return None
    
    def _calculate_session_statistics(self, session: DetectionSession) -> Dict[str, Any]:
        """Calculate statistics for a session."""
        
        completed_results = [r for r in session.results if r.status == "completed"]
        
        total_healthy = sum(r.healthy_count for r in completed_results)
        total_unhealthy = sum(r.unhealthy_count for r in completed_results)
        total_leaves = total_healthy + total_unhealthy
        
        return {
            'total_leaves': total_leaves,
            'total_healthy': total_healthy,
            'total_unhealthy': total_unhealthy,
            'healthy_percentage': (total_healthy / total_leaves * 100) if total_leaves > 0 else 0,
            'unhealthy_percentage': (total_unhealthy / total_leaves * 100) if total_leaves > 0 else 0,
            'average_health_percentage': sum(r.health_percentage for r in completed_results) / len(completed_results) if completed_results else 0,
            'average_processing_time': sum(r.processing_time for r in completed_results) / len(completed_results) if completed_results else 0,
            'completed_images': len(completed_results),
            'failed_images': len([r for r in session.results if r.status == "failed"])
        }
    
    def _generate_charts(self, session: DetectionSession, stats: Dict[str, Any]) -> List[str]:
        """Generate charts for the report."""
        
        chart_paths = []
        
        try:
            # Chart 1: Health distribution pie chart
            if stats['total_leaves'] > 0:
                plt.figure(figsize=(8, 6))
                sizes = [stats['total_healthy'], stats['total_unhealthy']]
                labels = ['Healthy', 'Unhealthy']
                colors_pie = ['#2ecc71', '#e74c3c']
                
                plt.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
                plt.title('Tea Leaf Health Distribution', fontsize=16, fontweight='bold')
                
                chart1_path = os.path.join(self.reports_dir, f"chart_health_dist_{session.id}.png")
                plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths.append(chart1_path)
            
            # Chart 2: Processing time vs image index
            if len(session.results) > 1:
                plt.figure(figsize=(10, 6))
                completed_results = [r for r in session.results if r.status == "completed"]
                
                if completed_results:
                    processing_times = [r.processing_time for r in completed_results]
                    indices = range(1, len(processing_times) + 1)
                    
                    plt.plot(indices, processing_times, marker='o', linewidth=2, markersize=4)
                    plt.title('Processing Time per Image', fontsize=16, fontweight='bold')
                    plt.xlabel('Image Number')
                    plt.ylabel('Processing Time (seconds)')
                    plt.grid(True, alpha=0.3)
                    
                    chart2_path = os.path.join(self.reports_dir, f"chart_processing_time_{session.id}.png")
                    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_paths.append(chart2_path)
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return chart_paths
    
    def get_session_summary(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get a summary of session results for quick overview."""
        
        session = self.db.query(DetectionSession).filter(
            DetectionSession.id == session_id
        ).first()
        
        if not session:
            return None
        
        stats = self._calculate_session_statistics(session)
        
        return {
            'session_id': session.id,
            'session_name': session.name,
            'status': session.status,
            'created_at': session.created_at,
            'total_images': session.total_images,
            'processed_images': session.processed_images,
            **stats
        }