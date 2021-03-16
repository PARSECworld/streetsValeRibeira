const {google} = require('googleapis');

var streetViewService = new google.maps.StreetViewService();
var STREETVIEW_MAX_DISTANCE = 100;

var [minx, miny, maxx, maxy] = [-48.851506, -24.514689, -48.846841999999995, -24.511943];
var N = 20;
var dx = (maxx-minx)/N;
var dy = (maxy-miny)/N;

for (var i = 0; i <= N; i++) {
    for (var j = 0; j <= N; j++) {
        let latLng = new google.maps.LatLng(minx + dx * i, miny + dy * j);
        streetViewService.getPanoramaByLocation(latLng, STREETVIEW_MAX_DISTANCE, function (streetViewPanoramaData, status) {
            if (status === google.maps.StreetViewStatus.OK) {
                console.log('Found');
            } else {
                console.log('-');
            }
        })
    }
}