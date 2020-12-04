import {Component, ElementRef, OnInit, ViewChild} from '@angular/core';
import { HomeService } from './services/home.service';

class ImageSnippet {
  constructor(public src: string, public file: File) { }
}
@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {
  selectedFile: any;
  public isSmile:boolean = false
  public isSad:boolean = false;
  public isNatural:boolean = false;
  public filePath="";
  public emojiPath = "assets/emoji/find.svg";
  @ViewChild('audioContainer', { static: true }) audioContainerRef: ElementRef;
  audioinput;
  public emotion='';
  public speech='';


  constructor(private homeService: HomeService) { }

  ngOnInit() {
  }


  processFile() {
    this.homeService.getEmotion(this.filePath)
      .subscribe(
        (res) => {
          this.setSmily(res[0])
          console.log(res);
        },
        (err) => {

          console.log(err);
        })
  }

  getSpeechToText() {
    this.homeService.getSpeech(this.filePath)
      .subscribe(
        (res) => {
          if(res){
          this.speech = res + '';
          console.log(res);
          }
          else{
            this.speech = '';
          }
        },
        (err) => {

          console.log(err);
        })
  }

  uploadFile(input: any) {
    this.setSmily('find')
    const file: File = input.files[0];
    this.homeService.uploadFile(file)
      .subscribe(
        (res) => {
          this.filePath = file.name;
          this.playAudio(this.audioFileToPlay);
          console.log(res);
        },
        (err) => {
          console.log(err);
        })
  }

  playAudio(file){
    let audio = new Audio();
    audio.src = file;
    audio.controls = true;
    audio.load();
    this.audioContainerRef.nativeElement.innerHTML = '';
    this.audioContainerRef.nativeElement.appendChild(audio);
    // audio.play();
  }

  public setSmily(em){
    if(em === 'find'){
      this.emotion='';
    }else{
      this.emotion=em;
    }
    this.emojiPath=`assets/emoji/${em}.svg`;
  }

  get audioFileToPlay() {
    return this.filePath ? `http://localhost:5000/playSoundFile/${this.filePath}` : '';
  }

  public getFullTxtResult(data){
    let res = '';
    if(data && data.alternative){
      data.alternative.forEach(element => {
       res +=  element.transcript + ' ';
      });
    }
    return res;
  }
}
