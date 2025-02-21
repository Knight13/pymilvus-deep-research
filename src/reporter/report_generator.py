class ReportGenerator:
    def __init__(self, topic: str, breakdown: dict, answers: dict):
        self.topic = topic
        self.breakdown = breakdown
        self.answers = answers

    def generate_markdown(self) -> str:
        report = [f'# {self.topic}\n\n']
        for key, sub_questions in self.breakdown.items():
            report.append(f'## {key}\n')
            if not sub_questions:
                report.append(self.answers.get(key, "") + '\n\n')
            else:
                for sub_q in sub_questions:
                    report.append(f'### {sub_q}\n')
                    report.append(self.answers.get(sub_q, "") + '\n\n')
        return "".join(report)
