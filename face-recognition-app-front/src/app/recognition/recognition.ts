import { Component } from '@angular/core';
import {HttpClient, HttpClientModule, provideHttpClient, withFetch, withInterceptorsFromDi} from '@angular/common/http';
import {CommonModule} from '@angular/common';

interface classItem {
  class: string;
  probability: number;
}

@Component({
  standalone: true,
  imports: [CommonModule],
  selector: 'app-recognition',
  templateUrl: './recognition.html',
  styleUrls: ['./recognition.scss']
})
export class RecognitionComponent {
  selectedFile: File | null = null;
  previewUrl: string | ArrayBuffer | null = null;
  result: string | null = null;
  probability: number | null = null;

  top5: classItem[] = [];

  faces = [
    '246.jpg',
    '4556254.jpg',
    '5812428.jpg',
    '4506699.jpg'
  ];

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];

      const reader = new FileReader();
      reader.onload = () => this.previewUrl = reader.result;
      reader.readAsDataURL(this.selectedFile);
    }
  }

  sendImage(path: string) {
    fetch(path)
      .then(res => res.blob())
      .then(blob => {
        const formData = new FormData();
        formData.append('file', blob, path.split('/').pop()!);

        this.http.post('http://localhost:8000/predict', formData).subscribe(
          (res: any) => this.top5 = res.top5_predictions,
          err => console.error(err)
        );
      });
  }

  sendToAPI(): void {
    if (!this.selectedFile) return;

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.http.post<any>('http://localhost:8000/predict', formData).subscribe(
      (res) => {
        console.log(res.top5_predictions)
        this.top5 = res.top5_predictions;
        console.log(this.top5)
        this.result = res.class;
        this.probability = res.probability;
      },
      (err) => {
        console.error(err);
        this.result = 'Erro na classificação';
      }
    );
  }
}
