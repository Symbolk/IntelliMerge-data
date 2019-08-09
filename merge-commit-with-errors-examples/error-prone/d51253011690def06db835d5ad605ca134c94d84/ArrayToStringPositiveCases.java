

package com.google.errorprone.bugpatterns;

import java.util.*;


public class ArrayToStringPositiveCases {

  public void intArray() {
    int[] a = {1, 2, 3};


    if (a.toString().isEmpty()) {
      System.out.println("int array string is empty!");
    } else {
      System.out.println("int array string is nonempty!");
    }
  }

  public void objectArray() {
    Object[] a = new Object[3];


    if (a.toString().isEmpty()) {
      System.out.println("object array string is empty!");
    } else {
      System.out.println("object array string is nonempty!");
    }
  }

  public void firstMethodCall() {
    String s = "hello";


    if (s.toCharArray().toString().isEmpty()) {
      System.out.println("char array string is empty!");
    } else {
      System.out.println("char array string is nonempty!");
    }
  }

  public void secondMethodCall() {
    char[] a = new char[3];


    if (a.toString().isEmpty()) {
      System.out.println("array string is empty!");
    } else {
      System.out.println("array string is nonempty!");
    }
  }
<<<<<<< HEAD
}
=======
}
>>>>>>> a90b6a7ce9d496a4ebc5c83e29c32c557a19a67d
