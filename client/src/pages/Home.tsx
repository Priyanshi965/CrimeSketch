import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, Upload, Zap, Search, BarChart3, History } from "lucide-react";
import { Link } from "wouter";

export default function Home() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-6 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="w-8 h-8 text-accent" />
            <h1 className="text-2xl font-bold tracking-tight">CrimeSketch AI</h1>
          </div>
          <div>
            <Button variant="outline" asChild>
              <Link href="/search">Search</Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="border-b border-border bg-gradient-to-b from-card to-background py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
              Intelligent Sketch-to-Face Recognition
            </h2>
            <p className="text-lg text-muted-foreground mb-8">
              Advanced AI-powered forensic analysis system for matching hand-drawn or uploaded suspect sketches against a comprehensive database of face images using deep learning and similarity search.
            </p>
            <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground" asChild>
              <Link href="/search">Get Started</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-4">
          <h3 className="text-3xl font-bold tracking-tight mb-12 text-center">Core Capabilities</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <Upload className="w-8 h-8 text-accent mb-2" />
                <CardTitle>Sketch Upload & Canvas</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Upload hand-drawn sketches or draw directly on our forensic canvas. Supports multiple image formats for maximum flexibility.
                </CardDescription>
              </CardContent>
            </Card>

            {/* Feature 2 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <Zap className="w-8 h-8 text-accent mb-2" />
                <CardTitle>AI-Powered Matching</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  ResNet50-based Siamese networks extract discriminative embeddings for accurate sketch-to-photo matching with confidence scores.
                </CardDescription>
              </CardContent>
            </Card>

            {/* Feature 3 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <Search className="w-8 h-8 text-accent mb-2" />
                <CardTitle>FAISS Similarity Search</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Lightning-fast similarity search against thousands of indexed face embeddings. Get top-K matches in milliseconds.
                </CardDescription>
              </CardContent>
            </Card>

            {/* Feature 4 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <BarChart3 className="w-8 h-8 text-accent mb-2" />
                <CardTitle>Pipeline Visualization</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Watch the complete recognition pipeline: upload → preprocess → extract → search → results. Full transparency in every step.
                </CardDescription>
              </CardContent>
            </Card>

            {/* Feature 5 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <History className="w-8 h-8 text-accent mb-2" />
                <CardTitle>Search History</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Track all past queries with timestamps and matched results. Comprehensive audit trail for forensic investigations.
                </CardDescription>
              </CardContent>
            </Card>

            {/* Feature 6 */}
            <Card className="bg-card border-border hover:border-accent transition-colors">
              <CardHeader>
                <Shield className="w-8 h-8 text-accent mb-2" />
                <CardTitle>Admin Dashboard</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Manage suspect database, view statistics, and re-index embeddings. Full control over the forensic system.
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t border-border bg-card py-16 md:py-24">
        <div className="container mx-auto px-4 text-center">
          <h3 className="text-3xl font-bold tracking-tight mb-6">Ready to Enhance Your Investigation?</h3>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            CrimeSketch AI combines cutting-edge deep learning with forensic expertise to help law enforcement agencies identify suspects faster and more accurately.
          </p>
          <Button size="lg" className="bg-accent hover:bg-accent/90 text-accent-foreground" asChild>
            <Link href="/search">Begin Investigation</Link>
          </Button>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-background py-8">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p>&copy; 2026 CrimeSketch AI. Advanced Forensic Recognition System.</p>
        </div>
      </footer>
    </div>
  );
}
