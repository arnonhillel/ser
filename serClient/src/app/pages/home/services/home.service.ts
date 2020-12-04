import { Injectable } from '@angular/core';
import { HttpHeaders, HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class HomeService {
  private serverUrl = environment.serverUrl;
  constructor(private http: HttpClient) {   
  }


  getEmotion(filePath:string){
    const url = `${this.serverUrl}/classifyEmotion?filename=${filePath}`;
    const httpOptions = {
      headers: new HttpHeaders({
      }),
    };
    return this.http.get(url,httpOptions).pipe(
      map((data) => {
        if (data) {
          return data;
        }
      })
    );

  }

  getSpeech(filePath:string){
    const url = `${this.serverUrl}/speechToText?filename=${filePath}`;
    const httpOptions = {
      headers: new HttpHeaders({
      }),
    };
    return this.http.get(url,httpOptions).pipe(
      map((data) => {
        if (data) {
          return data;
        }
      })
    );

  }



  uploadFile(filePath){
    const url = `${this.serverUrl}/uploader?filename=${filePath}`;
    var form = new FormData();
    form.append("file", filePath);
    form.append("", "");
    const httpOptions = {
      headers: new HttpHeaders({
      }),
    };
    return this.http.post(url,form,httpOptions).pipe(
      map((data) => {
        if (data) {
          return data;
        }
      })
    );

  }
}
