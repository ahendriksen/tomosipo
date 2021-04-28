Conventions
===========







Naming and ordering
----------------------------------

The following terms are used as parameters, properties, and in
function names:

-   **shape:** The number of voxels / pixels in each dimension
-   **size:** The physical size of the object in each dimension
-   **dist:** Short for distance
-   **vec:** Short for vector
-   **obj:** Short for object
-   **vol:** Short for volume
-   **pos:** Short for position: this is alway the **center** of an object
-   **src:** Short for source
-   **det:** Short for detector
-   **pos:** Short for position
-   **len:** Short for length
-   **rel:** Short for relative
-   **abs:** Short for absolute
-   **num_\*:** Number of \* (angles for instance)

In examples and code, projection and volume geometries are often
abbreviated as:

-   **pg:** projection geometry
-   **vg:** volume geometry

Whenever a function takes as parameters a volume geometry,
projection geometry, or operator, there is a fixed ordering:

1.  operator
2.  volume (data or geometry)
3.  projection (data or geometry)
