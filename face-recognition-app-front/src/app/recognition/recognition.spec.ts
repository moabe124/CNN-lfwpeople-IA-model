import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Recognition } from './recognition';

describe('Recognition', () => {
  let component: Recognition;
  let fixture: ComponentFixture<Recognition>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Recognition]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Recognition);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
