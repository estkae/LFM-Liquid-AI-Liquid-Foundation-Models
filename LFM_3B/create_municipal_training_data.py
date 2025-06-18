#!/usr/bin/env python3
"""
Create German municipal administration training data
"""

import json
import random


def create_municipal_training_data(output_file: str = "municipal_training_data.jsonl"):
    """Create training data for German municipal administration"""
    
    # Training examples for different departments
    training_examples = [
        # Einwohnermeldeamt
        {
            "text": "Frage: Ich möchte meinen Wohnsitz ummelden. Was muss ich tun?\nAntwort: Für die Ummeldung Ihres Wohnsitzes müssen Sie persönlich beim Einwohnermeldeamt erscheinen. Bringen Sie bitte folgende Unterlagen mit: Personalausweis oder Reisepass, Wohnungsgeberbestätigung vom Vermieter. Die Ummeldung muss innerhalb von 14 Tagen nach dem Umzug erfolgen. Die Gebühr beträgt 0 Euro.",
            "department": "einwohnermeldeamt"
        },
        {
            "text": "Frage: Wie beantrage ich eine Meldebescheinigung?\nAntwort: Eine Meldebescheinigung können Sie persönlich beim Einwohnermeldeamt beantragen. Benötigte Unterlagen: Personalausweis oder Reisepass. Die Gebühr beträgt 5 Euro. Die Bescheinigung wird sofort ausgestellt.",
            "department": "einwohnermeldeamt"
        },
        {
            "text": "Frage: Ich ziehe aus Deutschland weg. Muss ich mich abmelden?\nAntwort: Ja, bei einem Wegzug ins Ausland müssen Sie sich beim Einwohnermeldeamt abmelden. Die Abmeldung ist kostenlos und Sie erhalten eine Abmeldebescheinigung.",
            "department": "einwohnermeldeamt"
        },
        
        # Bauamt
        {
            "text": "Frage: Ich möchte einen Wintergarten anbauen. Brauche ich eine Genehmigung?\nAntwort: Für den Anbau eines Wintergartens benötigen Sie in der Regel eine Baugenehmigung. Reichen Sie beim Bauamt einen Bauantrag mit folgenden Unterlagen ein: Bauzeichnungen, Lageplan, Baubeschreibung, statische Berechnungen. Die Bearbeitungszeit beträgt etwa 3 Monate.",
            "department": "bauamt"
        },
        {
            "text": "Frage: Wie hoch sind die Gebühren für eine Baugenehmigung?\nAntwort: Die Gebühren für eine Baugenehmigung richten sich nach den Baukosten. Als Faustregel gilt: etwa 0,5% bis 1% der Baukosten. Die genaue Berechnung erfolgt nach der Gebührenordnung.",
            "department": "bauamt"
        },
        {
            "text": "Frage: Kann ich ohne Genehmigung einen Gartenhaus aufstellen?\nAntwort: Gartenhäuser bis 30 Kubikmeter umbauten Raum sind oft genehmigungsfrei, müssen aber trotzdem baurechtliche Vorschriften einhalten. Informieren Sie sich beim Bauamt über die lokalen Regelungen.",
            "department": "bauamt"
        },
        
        # Standesamt
        {
            "text": "Frage: Wie beantrage ich eine Geburtsurkunde?\nAntwort: Geburtsurkunden können Sie beim Standesamt des Geburtsortes beantragen. Möglich ist dies persönlich, schriftlich oder online. Benötigt werden: Personalausweis und Angaben zur Person. Gebühr: 12 Euro pro Urkunde.",
            "department": "standesamt"
        },
        {
            "text": "Frage: Welche Unterlagen brauche ich für die Eheschließung?\nAntwort: Für die Anmeldung zur Eheschließung benötigen Sie: Personalausweis oder Reisepass, beglaubigte Abschrift aus dem Geburtenregister (nicht älter als 6 Monate), Aufenthaltsbescheinigung. Bei Geschiedenen zusätzlich: rechtskräftiges Scheidungsurteil.",
            "department": "standesamt"
        },
        {
            "text": "Frage: Wie lange dauert es, eine Sterbeurkunde zu bekommen?\nAntwort: Sterbeurkunden werden in der Regel sofort ausgestellt, wenn alle erforderlichen Unterlagen vorliegen. Die Gebühr beträgt 12 Euro pro Urkunde.",
            "department": "standesamt"
        },
        
        # Ordnungsamt
        {
            "text": "Frage: Ich möchte mich über Lärmbelästigung beschweren. An wen wende ich mich?\nAntwort: Bei Lärmbelästigung ist das Ordnungsamt zuständig. Sie können eine Beschwerde schriftlich einreichen oder persönlich vorsprechen. Dokumentieren Sie die Störungen mit Datum und Uhrzeit. Das Ordnungsamt wird den Fall prüfen und gegebenenfalls Maßnahmen ergreifen.",
            "department": "ordnungsamt"
        },
        {
            "text": "Frage: Wie beantrage ich eine Sondernutzungserlaubnis für einen Umzug?\nAntwort: Für Umzugsfahrzeuge auf öffentlichen Straßen benötigen Sie eine Sondernutzungserlaubnis vom Ordnungsamt. Antrag mindestens 5 Werktage vorher stellen. Gebühr: ca. 30-50 Euro pro Tag.",
            "department": "ordnungsamt"
        },
        {
            "text": "Frage: Mein Nachbar parkt ständig vor meiner Einfahrt. Was kann ich tun?\nAntwort: Das Zuparken von Einfahrten ist eine Ordnungswidrigkeit. Melden Sie dies dem Ordnungsamt, das ein Bußgeld verhängen und das Fahrzeug abschleppen lassen kann.",
            "department": "ordnungsamt"
        },
        
        # Stadtkasse
        {
            "text": "Frage: Wie kann ich meine Grundsteuer bezahlen?\nAntwort: Die Grundsteuer können Sie per Überweisung, SEPA-Lastschrift oder bar in der Stadtkasse bezahlen. Bei Lastschrift erfolgt die Abbuchung automatisch zu den Fälligkeitsterminen.",
            "department": "stadtkasse"
        },
        {
            "text": "Frage: Ich habe eine Mahnung erhalten. Was soll ich tun?\nAntwort: Prüfen Sie zunächst, ob die Zahlung bereits erfolgt ist. Falls nicht, zahlen Sie den Betrag umgehend. Bei Zahlungsschwierigkeiten kontaktieren Sie die Stadtkasse für eine Ratenzahlung.",
            "department": "stadtkasse"
        },
        {
            "text": "Frage: Wo finde ich meine Steuernummer?\nAntwort: Ihre gemeindliche Steuernummer finden Sie auf dem Grundsteuerbescheid oder anderen Bescheiden der Stadt. Bei Fragen hilft Ihnen die Stadtkasse unter Angabe Ihrer Adresse.",
            "department": "stadtkasse"
        },
        
        # Sozialamt
        {
            "text": "Frage: Wie beantrage ich Wohngeld?\nAntwort: Wohngeld beantragen Sie beim Sozialamt mit dem Antragsformular. Benötigte Unterlagen: Einkommensnachweise, Mietvertrag, Kontoauszüge. Die Bearbeitung dauert etwa 4-6 Wochen.",
            "department": "sozialamt"
        },
        {
            "text": "Frage: Wer hat Anspruch auf Grundsicherung im Alter?\nAntwort: Grundsicherung erhalten Personen ab 67 Jahren, deren Einkommen und Vermögen nicht für den Lebensunterhalt ausreicht. Der Antrag ist beim Sozialamt zu stellen.",
            "department": "sozialamt"
        },
        {
            "text": "Frage: Gibt es Unterstützung für die Erstausstattung einer Wohnung?\nAntwort: Ja, bei Bedürftigkeit kann das Sozialamt Beihilfen für die Erstausstattung gewähren. Dies umfasst Möbel und Haushaltsgeräte. Ein begründeter Antrag mit Kostenvoranschlägen ist erforderlich.",
            "department": "sozialamt"
        },
        
        # Jugendamt
        {
            "text": "Frage: Wie beantrage ich einen Kita-Platz?\nAntwort: Kita-Plätze beantragen Sie über das Online-Portal der Stadt oder direkt beim Jugendamt. Die Anmeldung ist ab Geburt möglich. Berücksichtigt werden Berufstätigkeit der Eltern und soziale Kriterien.",
            "department": "jugendamt"
        },
        {
            "text": "Frage: Welche Unterlagen brauche ich für Elterngeld?\nAntwort: Für Elterngeld benötigen Sie: Geburtsurkunde des Kindes, Einkommensnachweise, Bescheinigung der Krankenkasse über Mutterschaftsgeld, Arbeitgeberbescheinigung. Der Antrag ist bei der Elterngeldstelle einzureichen.",
            "department": "jugendamt"
        },
        {
            "text": "Frage: Gibt es Ferienbetreuung für Schulkinder?\nAntwort: Ja, das Jugendamt organisiert Ferienbetreuung für Schulkinder. Anmeldung erfolgt online, die Plätze sind begrenzt. Kosten richten sich nach dem Einkommen der Eltern.",
            "department": "jugendamt"
        },
        
        # Allgemeine Verwaltung
        {
            "text": "Frage: Wo finde ich Formulare zum Download?\nAntwort: Alle Formulare finden Sie auf der Webseite der Stadt unter 'Service/Formulare'. Alternativ erhalten Sie diese im Bürgerbüro oder bei der zuständigen Behörde.",
            "department": "general"
        },
        {
            "text": "Frage: Wie sind die Öffnungszeiten des Rathauses?\nAntwort: Das Rathaus ist geöffnet: Montag bis Mittwoch 8-16 Uhr, Donnerstag 8-18 Uhr, Freitag 8-12 Uhr. Termine außerhalb der Öffnungszeiten nach Vereinbarung.",
            "department": "general"
        },
        {
            "text": "Frage: Kann ich online einen Termin vereinbaren?\nAntwort: Ja, über das Online-Terminvergabesystem der Stadt können Sie für viele Dienstleistungen Termine buchen. Wählen Sie die gewünschte Dienstleistung und einen freien Termin.",
            "department": "general"
        }
    ]
    
    # Additional dialog patterns
    dialog_templates = [
        "Bürger: {question}\nSachbearbeiter: {answer}",
        "Anfrage: {question}\nAuskunft: {answer}",
        "Kunde: {question}\nMitarbeiter: {answer}",
        "{question}\n{answer}"
    ]
    
    # Generate variations
    all_examples = []
    
    for example in training_examples:
        # Original example
        all_examples.append({
            "text": example["text"],
            "metadata": {"department": example["department"]}
        })
        
        # Create variations with different templates
        if "Frage:" in example["text"] and "Antwort:" in example["text"]:
            parts = example["text"].split("\n")
            question = parts[0].replace("Frage: ", "")
            answer = parts[1].replace("Antwort: ", "")
            
            for template in dialog_templates[:2]:  # Use first 2 templates for variation
                text = template.format(question=question, answer=answer)
                all_examples.append({
                    "text": text,
                    "metadata": {"department": example["department"]}
                })
    
    # Shuffle examples
    random.shuffle(all_examples)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(all_examples)} training examples in {output_file}")
    
    # Print statistics
    dept_counts = {}
    for ex in all_examples:
        dept = ex["metadata"]["department"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
    
    print("\n📊 Training data statistics:")
    for dept, count in sorted(dept_counts.items()):
        print(f"  {dept}: {count} examples")
    
    return output_file


if __name__ == "__main__":
    create_municipal_training_data()