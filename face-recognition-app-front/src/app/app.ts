import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import {RecognitionComponent} from './recognition/recognition';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RecognitionComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('face-recognition-app-front');
}
